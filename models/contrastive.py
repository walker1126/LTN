#!/usr/bin/env python3

"""Contrastive model."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import slowfast.models.losses as losses
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models.video_model_builder import X3D, MViT, ResNet, SlowFast

from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)

# Supported model types
_MODEL_TYPES = {
    "slowfast": SlowFast,
    "slow": ResNet,
    "c2d": ResNet,
    "i3d": ResNet,
    "slow_c2d": ResNet,
    "x3d": X3D,
    "mvit": MViT,
}



@MODEL_REGISTRY.register()
class ContrastiveModel(nn.Module):
    """
    Contrastive Model, currently mainly focused on memory bank and CSC.
    """

    def __init__(self, cfg):
        super(ContrastiveModel, self).__init__()
        # Construct the model.
        self.backbone = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
        self.type = cfg.CONTRASTIVE.TYPE
        self.T = cfg.CONTRASTIVE.T
        self.dim = cfg.CONTRASTIVE.DIM
        self.length = cfg.CONTRASTIVE.LENGTH
        self.k = cfg.CONTRASTIVE.QUEUE_LEN
        self.mmt = cfg.CONTRASTIVE.MOMENTUM
        self.momentum_annealing = cfg.CONTRASTIVE.MOMENTUM_ANNEALING
        self.duration = 1
        self.cfg = cfg
        self.num_gpus = cfg.NUM_GPUS
        self.l2_norm = Normalize()
        self.knn_num_imgs = 0
        self.knn_on = cfg.CONTRASTIVE.KNN_ON
        self.train_labels = np.zeros((0,), dtype=np.int32)
        self.num_pos = 2
        self.num_crops = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            * self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL
        )
        self.nce_loss_fun = losses.get_loss_func("contrastive_loss")(
            reduction="mean"
        )
        assert self.cfg.MODEL.LOSS_FUNC == "contrastive_loss"
        self.softmax = nn.Softmax(dim=1).cuda()

        if self.type == "moco" or self.type == "byol":
            # MoCo components
            self.backbone_hist = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
            for p in self.backbone_hist.parameters():
                p.requires_grad = False
            self.register_buffer("ptr", torch.tensor([0]))
            self.ptr.requires_grad = False
            stdv = 1.0 / math.sqrt(self.dim / 3)
            self.register_buffer(
                "queue_x",
                torch.rand(self.k, self.dim).mul_(2 * stdv).add_(-stdv),
            )
            self.register_buffer("iter", torch.zeros([1], dtype=torch.long))
            self._batch_shuffle_on = (
                False
                if (
                    "sync" in cfg.BN.NORM_TYPE
                    and cfg.BN.NUM_SYNC_DEVICES == cfg.NUM_GPUS
                )
                or self.type == "byol"
                else True
            )
        if self.knn_on:
            self.knn_mem = Memory(self.length, 1, self.dim, cfg)

    @torch.no_grad()
    def knn_mem_update(self, q_knn, index):
        if self.knn_on:
            self.knn_mem.update(
                q_knn,
                momentum=1.0,
                ind=index,
                time=torch.zeros_like(index),
                interp=False,
            )

    @torch.no_grad()
    def init_knn_labels(self, train_loader):
        logger.info("initializing knn labels")
        self.num_imgs = len(train_loader.dataset._labels)
        self.train_labels = np.zeros((self.num_imgs,), dtype=np.int32)
        for i in range(self.num_imgs):
            self.train_labels[i] = train_loader.dataset._labels[i]
        self.train_labels = torch.LongTensor(self.train_labels).cuda()
        if self.length != self.num_imgs:
            logger.error(
                "Kinetics dataloader size: {} differs from memorybank length {}".format(
                    self.num_imgs, self.length
                )
            )
            self.knn_mem.resize(self.num_imgs, 1, self.dim)

    @torch.no_grad()
    def _update_history(self):
        # momentum update
        iter = int(self.iter)
        m = self.mmt
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p

        if iter == 0:
            for name, p in self.backbone_hist.named_parameters():
                p.data.copy_(dist[name].data)

        for name, p in self.backbone_hist.named_parameters():
            p.data = dist[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def _batch_shuffle(self, x, dt=None):
        if len(x) == 2:
            another_crop = True
        else:
            another_crop = False
        if another_crop:
            x, x_crop = x[0], x[1]
        else:
            x = x[0]

        world_size = self.cfg.NUM_GPUS * self.cfg.NUM_SHARDS
        if self.num_gpus > 1:
            if self.cfg.CONTRASTIVE.LOCAL_SHUFFLE_BN:
                x = du.cat_all_gather(x, local=True)
                if another_crop:
                    x_crop = du.cat_all_gather(x_crop, local=True)
                world_size = du.get_local_size()
                gpu_idx = du.get_local_rank()
            else:
                x = du.cat_all_gather(x)
                if another_crop:
                    x_crop = du.cat_all_gather(x_crop)
                gpu_idx = torch.distributed.get_rank()

        idx_randperm = torch.randperm(x.shape[0]).cuda()
        if self.num_gpus > 1:
            torch.distributed.broadcast(idx_randperm, src=0)
        else:
            gpu_idx = 0
        idx_randperm = idx_randperm.view(world_size, -1)
        x = x[idx_randperm[gpu_idx, :]]
        if another_crop:
            x_crop = x_crop[idx_randperm[gpu_idx, :]]

        idx_restore = torch.argsort(idx_randperm.view(-1))
        idx_restore = idx_restore.view(world_size, -1)
        if dt is None:
            if another_crop:
                return [x, x_crop], idx_restore
            else:
                return [x], idx_restore
        else:
            if self.num_gpus > 1:
                dt = du.cat_all_gather(dt.contiguous(), local=True)
            dt = dt[idx_randperm[gpu_idx, :]]
            if another_crop:
                return [x, x_crop], dt, idx_restore
            else:
                return [x], dt, idx_restore
       
    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_restore):
        if self.num_gpus > 1:
            if self.cfg.CONTRASTIVE.LOCAL_SHUFFLE_BN:
                x = du.cat_all_gather(x, local=True)
                gpu_idx = du.get_local_rank()
            else:
                x = du.cat_all_gather(x)
                gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0

        idx = idx_restore[gpu_idx, :]
        x = x[idx]
        return x

    @torch.no_grad()
    def eval_knn(self, q_knn, knn_k=200):
        with torch.no_grad():
            dist = torch.einsum(
                "nc,mc->nm",
                q_knn.view(q_knn.size(0), -1),
                self.knn_mem.memory.view(self.knn_mem.memory.size(0), -1),
            )
            yd, yi = dist.topk(knn_k, dim=1, largest=True, sorted=True)
        return yd, yi

    def sim_loss(self, q, k):
        similarity = torch.einsum("nc,nc->n", [q, k])  # N-dim
        # similarity += delta_t # higher if time distance is larger
        # sim = sim - max_margin + delta_t * k
        similarity /= self.T  # history-compatible
        loss = -similarity.mean()
        return loss

    @torch.no_grad()
    def momentum_anneal_cosine(self, epoch_exact):
        self.mmt = (
            1
            - (1 - self.cfg.CONTRASTIVE.MOMENTUM)
            * (
                math.cos(math.pi * epoch_exact / self.cfg.SOLVER.MAX_EPOCH)
                + 1.0
            )
            * 0.5
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, extra_keys=None):
        ptr = int(self.ptr.item())
        if (
            not self.cfg.CONTRASTIVE.MOCO_MULTI_VIEW_QUEUE
        ):
            keys_queue_update = [keys[0]]
        else:
            assert (
                len(keys) > 0
            ), "need to have multiple views for adding them to queue"
            keys_queue_update = []
            keys_queue_update += keys
            if extra_keys:
                keys_queue_update += [
                    item for sublist in extra_keys for item in sublist
                ]
        for key in keys_queue_update:
            # write the current feat into queue, at pointer
            num_items = int(key.size(0))

            assert self.k % num_items == 0
            assert ptr + num_items <= self.k
            self.queue_x[ptr : ptr + num_items, :] = key
            # move pointer
            ptr += num_items
            # reset pointer
            if ptr == self.k:
                ptr = 0
            self.ptr[0] = ptr

    @torch.no_grad()
    def batch_clips(self, clips):
        clips_batched = [None] * len(clips[0])
        for i, clip in enumerate(clips):
            for j, view in enumerate(clip):
                if i == 0:
                    clips_batched[j] = view
                else:
                    clips_batched[j] = torch.cat(
                        [clips_batched[j], view], dim=0
                    )
                del view
        return clips_batched

    @torch.no_grad()
    def compute_key_feat(
        self, clips_k, dt=None, compute_predictor_keys=False, batched_inference=True
    ):
        assert self.training
        # momentum update key encoder
        self._update_history()
        self.iter += 1
        n_clips = len(clips_k)
        bsz = clips_k[0][0].shape[0]
        if n_clips * bsz * clips_k[0][0].numel() > 4 * 64 * 3 * 8 * 224 * 224:
            batched_inference = False  # hack to avoid oom on large inputs
        assert n_clips > 0
        if batched_inference and all(
            [
                clips_k[i][j].shape[1:] == clips_k[0][j].shape[1:]
                for i in range(len(clips_k))
                for j in range(len(clips_k[i]))
            ]
        ):
            clips_k = [self.batch_clips(clips_k)]
            batched = True
        else:
            batched = False
        # LTN
        keys, pred_keys, hkeys = [], [], []
        for k in range(0, len(clips_k)):
            clip_k = clips_k[k]
            if self._batch_shuffle_on:
                with torch.no_grad():
                    if dt is None:
                        clip_k, idx_restore = self._batch_shuffle(clip_k)
                    else:
                        dt_k = dt[:, k, :]
                        clip_k, dt_k, idx_restore = self._batch_shuffle(clip_k, dt_k)
            with torch.no_grad():
                # LTN
                if dt is None:
                    hist_feat, hhist_feat = self.backbone_hist(clip_k, kl=True)
                else:
                    if self._batch_shuffle_on:
                        hist_feat, hhist_feat = self.backbone_hist(clip_k, dt_k, kl=True)
                    else:
                        hist_feat, hhist_feat = self.backbone_hist(clip_k, dt[:, k, :], kl=True)

                if isinstance(hist_feat, list):
                    hist_time = hist_feat[1:]
                    hist_feat = hist_feat[0]
                    if compute_predictor_keys:
                        tks = []
                        for tk in hist_time:
                            tk = self.l2_norm(tk)
                            if self._batch_shuffle_on:
                                tk = self._batch_unshuffle(
                                    tk, idx_restore
                                ).detach()
                            tks.append(tk)
                        pred_keys.append(tks)
                x_hist = self.l2_norm(hist_feat)
                if self._batch_shuffle_on:
                    x_hist = self._batch_unshuffle(x_hist, idx_restore).detach()
                    hhist_feat = self._batch_unshuffle(hhist_feat, idx_restore).detach()
            keys.append(x_hist)
            # LTN
            hkeys.append(hhist_feat)
        if batched:
            assert len(keys) == 1, "batched input uses single clip"
            batched_key = keys[0]
            if compute_predictor_keys:
                batched_pred_key = pred_keys[0]
            keys, pred_keys = [], []
            for k in range(0, n_clips):
                keys.append(batched_key[k * bsz : (k + 1) * bsz])
                if compute_predictor_keys:
                    pred_keys.append(batched_pred_key[k * bsz : (k + 1) * bsz])
        if compute_predictor_keys:
            return keys, pred_keys
        else:
            return [keys, hkeys]

    def forward(
        self, clips, index=None, time=None, epoch_exact=None, keys=None
    ):
        if epoch_exact is not None and self.momentum_annealing:
            self.momentum_anneal_cosine(epoch_exact)

        if self.type == "moco":
           
            if isinstance(clips[0], list):
                n_clips = len(clips)
                ind_clips = np.arange(
                    n_clips
                )  # clips come ordered temporally from decoder

                clip_q = clips[ind_clips[0]]
                clips_k = [clips[i] for i in ind_clips[1:]]
                # rearange time
                time_q = time[:, ind_clips[0], :]
                time_k = (
                    time[:, ind_clips[1:], :]
                    if keys is None
                    else time[:, ind_clips[0] + 1 :, :]
                )
            else:
                clip_q = clips
            
            # LTN
            if isinstance(clips[0], list)
                feat_q, hfeat_q = self.backbone(clip_q, time_q, True)
                hfeat_q = self.l2_norm(hfeat_q)
            else:
                feat_q = self.backbone(clip_q)

            extra_projs = []
            if isinstance(feat_q, list):
                extra_projs = feat_q[1:]
                feat_q = feat_q[0]
                extra_projs = [self.l2_norm(feat) for feat in extra_projs]

            if index is None:
                return feat_q
            
            q = self.l2_norm(feat_q)
            q_knn = q

            if not self.training:
                return self.eval_knn(q_knn)
        
            if keys is None:
                keys = self.compute_key_feat(
                    clips_k, time_k, compute_predictor_keys=False
                )
                auto_enqueue_keys = True
            else:
                auto_enqueue_keys = False
            # score computation
            queue_neg = torch.einsum(
                "nc,kc->nk", [q, self.queue_x.clone().detach()]
            )
        
            for k in range(len(keys[0])):
                if k == len(keys[0]):
                    key = extra_projs[0]
                    out_pos = torch.einsum("nc,nc->n", [q, key]).unsqueeze(-1)
                    queue_neg = torch.einsum(
                        "nc,kc->nk", [q, self.queue_x.clone().detach()]
                    )
                elif k == len(keys[0])-1:
                    key = keys[0][k]
                    hkey = self.l2_norm(keys[1][k])
                    out_pos = torch.einsum("nc,nc->n", [q, key]).unsqueeze(-1)
                else:
                    key = keys[0][k]
                    hkey = self.l2_norm(keys[1][k])
                    out_pos = torch.einsum("nc,nc->n", [q, key]).unsqueeze(-1)
                lgt_k = torch.cat([out_pos, queue_neg], dim=1)
                if k == 0:
                    logits = lgt_k
                else:
                    logits = torch.cat([logits, lgt_k], dim=0)

            logits = torch.div(logits, self.T)
            loss = self.nce_loss_fun(logits) 
            # update queue
            if self.training and auto_enqueue_keys:
                self._dequeue_and_enqueue(keys[0])

            self.knn_mem_update(q_knn, index)
            return logits, loss

        elif self.type == "byol":
            clips_key = [None] * len(clips)
            for i, clip in enumerate(clips):
                p = []
                for path in clip:
                    p.append(path)
                clips_key[i] = p
            batch_clips = False
            if isinstance(clips[0], list):
                n_clips = len(clips)
                ind_clips = np.arange(
                    n_clips
                )  # clips come ordered temporally from decoder

                # rearange time
                time_q = time[:, ind_clips[0], :]
                time_k = (
                    time[:, ind_clips[1:], :]
                    if keys is None
                    else time[:, ind_clips[0] + 1 :, :]
                )

                if batch_clips and n_clips > 1:
                    clips_batched = self.batch_clips(clips)
                    clips_key = [clips_batched]
                    clip_q = clips_batched
                else:
                    clip_q = clips[0]
                # LTN
                feat_q = self.backbone(clip_q, time_q)
            else:
                clip_q = clips
                feat_q = self.backbone(clip_q)

            predictors = []
            if isinstance(feat_q, list):
                predictors = feat_q[1:]
                feat_q = feat_q[0]
                predictors = [self.l2_norm(feat) for feat in predictors]
            else:
                raise NotImplementedError("BYOL: predictor is missing")
            assert len(predictors) == 1
            if index is None:
                return feat_q
            q = self.l2_norm(feat_q)

            q_knn = q  # projector

            if not self.training:
                return self.eval_knn(q_knn)

            ind_clips = np.arange(
                n_clips
            )  # clips come ordered temporally from decoder

            # rest down is for training
            if keys is None:
                keys = self.compute_key_feat(
                    clips_key, time_k, compute_predictor_keys=False
                )

            if self.cfg.CONTRASTIVE.SEQUENTIAL:
                loss_reg = self.sim_loss(predictors[0], keys[0][0])
                for i in range(1, len(keys[0])):
                    loss_reg += self.sim_loss(predictors[0], keys[0][i])
                loss_reg /= len(keys[0])
            else:
                if batch_clips:
                    bs = predictors[0].shape[0] // 2
                    loss_reg = self.sim_loss(
                        predictors[0][:bs, :], keys[0][0][bs:, :]
                    ) + self.sim_loss(predictors[0][bs:, :], keys[0][0][:bs, :])
                    q_knn = q_knn[:bs, :]
                    del clips_batched[0]
                else:
                    loss_q1 = self.sim_loss(predictors[0], keys[0][1])
                    assert len(clips) == 2
                    clip_q2 = clips[1]
                    feat_q2 = self.backbone(clip_q2)
                    predictors2 = feat_q2[1:]
                    predictors2 = [self.l2_norm(feat) for feat in predictors2]
                    assert len(predictors2) == 1

                    loss_q2 = self.sim_loss(predictors2[0], keys[0][0])
                    loss_reg = loss_q1 + loss_q2

            dummy_logits = torch.cat(
                (
                    9999.0
                    * torch.ones((len(index), 1), dtype=torch.float).cuda(),
                    torch.zeros((len(index), self.k), dtype=torch.float).cuda(),
                ),
                dim=1,
            )

            self.knn_mem_update(q_knn, index)

            return dummy_logits, loss_reg




    @torch.no_grad()
    def get_code(self, out):
        with torch.no_grad():
            Q = torch.exp(out / self.swav_eps_sinkhorn)  # BxK
            if self.cfg.NUM_SHARDS > 1:
                Q_sink = self.distributed_sinkhorn(Q.t(), 3)  # BxK
            else:
                Q_sink = self.sinkhorn(Q, 3)  # BxK
        return Q_sink


    @torch.no_grad()
    def sinkhorn(self, Q, iters):
        with torch.no_grad():
            Q = Q.t()  # KxB
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

            for _ in range(iters):
                Q *= (r / torch.sum(Q, dim=1)).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            return Q.t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            du.all_reduce([sum_Q], average=False)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (
                du.get_world_size() * Q.shape[1]
            )

            curr_sum = torch.sum(Q, dim=1)
            du.all_reduce([curr_sum], average=False)

            for _ in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                du.all_reduce([curr_sum], average=False)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def KLDivLoss(self, out, code):
        softmax = nn.Softmax(dim=1).cuda()
        p = softmax(out / self.T)
        loss = torch.mean(-torch.sum(code * torch.log(p), dim=1))
        return loss


def l2_loss(x, y):
    return 2 - 2 * (x * y).sum(dim=-1)


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.dim = dim
        self.power = power

    def forward(self, x):
        norm = (
            x.pow(self.power).sum(self.dim, keepdim=True).pow(1.0 / self.power)
        )
        out = x.div(norm)
        return out


class Memory(nn.Module):
    def __init__(self, length, duration, dim, cfg):
        super(Memory, self).__init__()
        self.length = length
        self.duration = duration
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer(
            "memory",
            torch.rand(length, duration, dim).mul_(2 * stdv).add_(-stdv),
        )
        self.device = self.memory.device
        self.l2_norm = Normalize(dim=1)
        self.l2_norm2d = Normalize(dim=2)
        self.num_gpus = cfg.NUM_GPUS

    def resize(self, length, duration, dim):
        self.length = length
        self.duration = duration
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        del self.memory
        self.memory = (
            torch.rand(length, duration, dim, device=self.device)
            .mul_(2 * stdv)
            .add_(-stdv)
            .cuda()
        )

    def get(self, ind, time, interp=False):
        batch_size = ind.size(0)
        with torch.no_grad():
            if interp:
                # mem_idx = self.memory[ind.view(-1), :, :]
                t0 = time.floor().long()  # - 1
                t0 = torch.clamp(t0, 0, self.memory.shape[1] - 1)
                t1 = t0 + 1
                t1 = torch.clamp(t1, 0, self.memory.shape[1] - 1)

                mem_t0 = self.memory[ind.view(-1), t0.view(-1), :]
                mem_t1 = self.memory[ind.view(-1), t1.view(-1), :]
                w2 = time.view(-1, 1) / self.duration
                w_t1 = (time - t0).view(-1, 1).float()
                w_t1 = 1 - w_t1  # hack for inverse
                selected_mem = mem_t0 * (1 - w_t1) + mem_t1 * w_t1
            else:
                # logger.info("1dmem get ind shape {} time shape {}".format(ind.shape, time.shape))
                selected_mem = self.memory[
                    ind.view(-1), time.long().view(-1), :
                ]

        out = selected_mem.view(batch_size, -1, self.dim)
        return out

    def update(self, mem, momentum, ind, time, interp=False):
        if self.num_gpus > 1:
            mem, ind, time = du.all_gather([mem, ind, time])
        with torch.no_grad():
            if interp:
                t0 = time.floor().long()  # - 1
                t0 = torch.clamp(t0, 0, self.memory.shape[1] - 1)
                t1 = t0 + 1
                t1 = torch.clamp(t1, 0, self.memory.shape[1] - 1)
                mem_t0 = self.memory[ind.view(-1), t0.view(-1), :]
                mem_t1 = self.memory[ind.view(-1), t1.view(-1), :]
                w2 = time.float().view(-1, 1) / float(self.duration)
                w_t1 = (time - t0).view(-1, 1).float()
                w_t1 = 1 - w_t1  # hack for inverse

                w_t0 = 1 - w_t1
                # mem = mem.squeeze()
                duo_update = False
                if duo_update:
                    update_t0 = (
                        mem * w_t0 + mem_t0 * w_t1
                    ) * momentum + mem_t0 * (1 - momentum)
                    update_t1 = (
                        mem * w_t1 + mem_t1 * w_t0
                    ) * momentum + mem_t1 * (1 - momentum)
                else:
                    update_t0 = mem * w_t0 * momentum + mem_t0 * (1 - momentum)
                    update_t1 = mem * w_t1 * momentum + mem_t1 * (1 - momentum)

                update_t0 = self.l2_norm(update_t0)
                update_t1 = self.l2_norm(update_t1)

                self.memory[ind.view(-1), t0.view(-1), :] = update_t0.squeeze()
                self.memory[ind.view(-1), t1.view(-1), :] = update_t1.squeeze()
            else:
                mem = mem.view(mem.size(0), 1, -1)
                mem_old = self.get(ind, time, interp=interp)
                mem_update = mem * momentum + mem_old * (1 - momentum)
                mem_update = self.l2_norm2d(mem_update)
                # logger.info("1dmem set ind shape {} time shape {}".format(ind.shape, time.shape))

                # my version
                self.memory[
                    ind.view(-1), time.long().view(-1), :
                ] = mem_update.squeeze()
                return

    def forward(self, inputs):
        pass

