#!/usr/bin/env python3

"""LTN Head helper."""
import numpy as np
import torch
import torch.nn as nn

#from ltn.models.batchnorm_helper import (
#    NaiveSyncBatchNorm1d as NaiveSyncBatchNorm1d,
#)
#from ltn.models.nonlocal_helper import Nonlocal



class MLPHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_dim,
        num_layers,
        bn_on=False,
        bias=True,
        flatten=False,
        xavier_init=True,
        bn_sync_num=1,
    ):
        super(MLPHead, self).__init__()
        self.flatten = flatten
        b = False if bn_on else bias
        # assert bn_on or bn_sync_num=1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
        mlp_layers[-1].xavier_init = xavier_init
        for i in range(1, num_layers):
            if bn_on:
                if bn_sync_num > 1:
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(
                            num_sync_devices=bn_sync_num, num_features=mlp_dim
                        )
                    )
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if i == num_layers - 1:
                d = dim_out
                b = bias
            else:
                d = mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
            mlp_layers[-1].xavier_init = xavier_init
        self.projection = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute((0, 2, 3, 4, 1))
        if self.flatten:
            x = x.reshape(-1, x.shape[-1])

        return self.projection(x)

class LTN(nn.Module):
    def __init__(self, size, dim, variant=3):
        super(Tbias, self).__init__()

        self.variant = variant
        self.size = size
        self.D = nn.Parameter(torch.Tensor(dim, size))
        nn.init.orthogonal_(self.D)

        xavier_init = False
        dimt = 2048
        mlp1 = nn.Linear(3, dimt)
        mlp1.xavier_init = xavier_init
#        mlp2 = nn.Linear(dimt, dimt)
#        mlp2.xavier_init = xavier_init
        mlp3 =  nn.Linear(dimt, size)
        mlp3.xavier_init = xavier_init

        self.MLP = nn.Sequential(nn.BatchNorm1d(3), mlp1, nn.ReLU(), mlp3)
        self.MLPx = nn.Sequential(nn.Linear(dim+size, dim+size), nn.ReLU(), nn.Linear(dim+size, size))
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, dt):
        if dt == None:
            return x

        dt = dt.view(dt.shape[0], -1)
        A = (self.MLP(dt.float())) # bs * dim
        weight = self.D + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if self.variant = 1:
            A = self.MLPx(torch.cat((A, x), dim=1))
            return x + A

        elif self.variant = 2:
            W = self.soft(self.MLPx(torch.cat((A, x), dim=1)))
            W = torch.matmul(A, Q.T)
            return x * W

        elif self.variant = 3:
            A = self.MLPx(torch.cat((A, x), dim=1))
            A = torch.matmul(A, Q.T)
            return x + A

        else:
            return x

class ResNetLTNHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        detach_final_fc=False,
        cfg=None,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            detach_final_fc (bool): if True, detach the fc layer from the
                gradient graph. By doing so, only the final fc layer will be
                trained.
            cfg (struct): The config for the current experiment.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc
        self.cfg = cfg
        self.local_projection_modules = []
        self.predictors = nn.ModuleList()
        self.l2norm_feats = False

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if cfg.CONTRASTIVE.NUM_MLP_LAYERS == 1:
            # LTN
            self.ltn = LTN(64, sum(dim_in), 3)
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        else:
            # LTN
            self.ltn = LTN(64, sum(dim_in), 3)
            self.projection = MLPHead(
                sum(dim_in),
                num_classes,
                cfg.CONTRASTIVE.MLP_DIM,
                cfg.CONTRASTIVE.NUM_MLP_LAYERS,
                bn_on=cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                if cfg.CONTRASTIVE.BN_SYNC_MLP
                else 1,
            )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

        if cfg.CONTRASTIVE.PREDICTOR_DEPTHS:
            d_in = num_classes
            for i, n_layers in enumerate(cfg.CONTRASTIVE.PREDICTOR_DEPTHS):
                local_mlp = MLPHead(
                    d_in,
                    num_classes,
                    cfg.CONTRASTIVE.MLP_DIM,
                    n_layers,
                    bn_on=cfg.CONTRASTIVE.BN_MLP,
                    flatten=False,
                    bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                    if cfg.CONTRASTIVE.BN_SYNC_MLP
                    else 1,
                )
                self.predictors.append(local_mlp)

    def forward(self, inputs, dt=None, kl=False):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        if self.l2norm_feats:
            x = nn.functional.normalize(x, dim=1, p=2)

        if (
            x.shape[1:4] == torch.Size([1, 1, 1])
            and self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        ):
            x = x.view(x.shape[0], -1)

        # LTN
        if dt is not None:
            x = self.ltn(x, dt)
            x_proj = self.projection(x)
        else:
            x_proj = self.projection(x)

        time_projs = []
        if self.predictors:
            x_in = x_proj
            for proj in self.predictors:
                time_projs.append(proj(x_in))

        if not self.training:
            if self.act is not None:
                x_proj = self.act(x_proj)
            # Performs fully convlutional inference.
            if x_proj.ndim == 5 and x_proj.shape[1:4] > torch.Size([1, 1, 1]):
                x_proj = x_proj.mean([1, 2, 3])

        x_proj = x_proj.view(x_proj.shape[0], -1)

        if kl:
            return x_proj, x.view(x.shape[0], -1)

        if time_projs:
            return [x_proj] + time_projs
        else:
            return x_proj

