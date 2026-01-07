import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.utils import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, act_cfg=None, norm_cfg=None):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # [N, C, H, W] -> [N, HW, C]
        n, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [N, HW, C]
        x = self.proj(x)
        return x


class LKA_Branch(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_cfg, act_cfg):
        super(LKA_Branch, self).__init__()

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 7, 1, 3, groups=in_channels)
        ])

        self.dw_d_conv = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=in_channels
        )

        self.pw_conv_att = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None  
        )

        self.pw_conv_proj = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        dw_outs = [conv(x) for conv in self.dw_convs]
        dw_avg = torch.mean(torch.stack(dw_outs, dim=0), dim=0)

        d_out = self.dw_d_conv(dw_avg)

        att = self.pw_conv_att(d_out)       # [N, C_in, H, W]
        att = torch.sigmoid(att)            # [0,1] gating

        modulated = att * x                 # [N, C_in, H, W]

        out = self.pw_conv_proj(modulated)  # [N, out_channels, H, W]
        return out


class LKA(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, norm_cfg, act_cfg):
        super(LKA, self).__init__()

        self.branches = nn.ModuleList([
            LKA_Branch(in_channels, out_channels, d, norm_cfg, act_cfg)
            for d in dilations
        ])

        self.bottleneck = ConvModule(
            len(dilations) * out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.bottleneck(out)
        return out


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    # elif type == 'conv':
    #     return ConvModule(in_channels, out_channels,
    #                       kernel_size=kwargs.get('kernel_size', 1),
    #                       padding=kwargs.get('kernel_size', 1)//2,
    #                       norm_cfg=kwargs.get('norm_cfg', None),
    #                       act_cfg=kwargs.get('act_cfg', None))
    elif type == 'lka':
        return LKA(in_channels, out_channels,
                   dilations=kwargs.get('dilations', (1, 6, 12, 18)),
                   norm_cfg=kwargs.get('norm_cfg', None),
                   act_cfg=kwargs.get('act_cfg', None))
    else:
        raise NotImplementedError(f"Unknown layer type: {type}")


@HEADS.register_module()
class ALKAHead(BaseDecodeHead):
    """
    ALKAHead: ASPP replaced with LKA (Large Kernel Attention) structure.
    Compatible with mmsegmentation 1.2.
    """

    def __init__(self, **kwargs):
        decoder_params = kwargs.pop('decoder_params', None)

        super(ALKAHead, self).__init__(input_transform='multiple_select', **kwargs)

        if decoder_params is not None:
            embed_cfg = decoder_params.get('embed_cfg', {})
            fusion_cfg = decoder_params.get('fusion_cfg', {})

            embed_dims = decoder_params.get('embed_dims', 256)
            if isinstance(embed_dims, int):
                embed_dims = [embed_dims] * len(self.in_index)

            self.embed_layers = nn.ModuleDict()
            for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
                self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)

            self.fuse_layer = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

        # decoder_params = kwargs['decoder_params']
        # embed_dims = decoder_params['embed_dims']
        # if isinstance(embed_dims, int):
        #     embed_dims = [embed_dims] * len(self.in_index)

        # embed_cfg = decoder_params['embed_cfg']
        # fusion_cfg = decoder_params['fusion_cfg']

        # self.embed_layers = nn.ModuleDict()
        # for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
        #     self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)

        # self.fuse_layer = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

    def forward(self, inputs):
        x = inputs
        n = x[0].shape[0]
        os_size = x[0].size()[2:]

        _c = {}
        for i in self.in_index:
            feat = self.embed_layers[str(i)](x[i])

            if feat.dim() == 3:
                feat = feat.permute(0, 2, 1).contiguous().reshape(
                    n, -1, x[i].shape[2], x[i].shape[3])
            if feat.size()[2:] != os_size:
                feat = resize(feat, size=os_size, mode='bilinear', align_corners=self.align_corners)
            _c[i] = feat

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)
        return x
