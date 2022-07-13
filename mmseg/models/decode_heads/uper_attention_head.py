# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmseg.ops import SwinBlockSequence
from mmcv.cnn import build_norm_layer


@HEADS.register_module()
class UPerAttentionHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 conv_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 **kwargs):
        super(UPerAttentionHead, self).__init__(
            input_transform='multiple_select', init_cfg=init_cfg, **kwargs)

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.bottleneck_attention = SwinBlockSequence(
                embed_dims=self.in_channels[3] + len(pool_scales) * self.channels,
                num_heads=num_heads[3],
                feedforward_channels=int(mlp_ratio * (self.in_channels[3] + len(pool_scales) * self.channels)),
                depth=depths[3],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                upsample=None,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                with_cp=False,
                init_cfg=None)

        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=self.act_cfg)

        self.psp_norm = build_norm_layer(norm_cfg, self.in_channels[-1] + len(pool_scales) * self.channels)[1]

        # FPN Module
        self.lateral_swins = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.lateral_norms = nn.ModuleList()

        for i in range(len(self.in_channels) - 1):  # skip the top layer
            l_swin = SwinBlockSequence(
                embed_dims=self.in_channels[i],
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * self.in_channels[i]),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                upsample=None,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=False,
                init_cfg=None)

            l_norm = build_norm_layer(norm_cfg, self.in_channels[i])[1]

            l_conv = ConvModule(
                self.in_channels[i],
                self.channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=act_cfg,
                inplace=False
            )

            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_swins.append(l_swin)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.lateral_norms.append(l_norm)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)

        hw_shape = psp_outs.shape[2:]
        psp_outs = psp_outs.flatten(2).transpose(1, 2)
        psp_outs = self.bottleneck_attention(psp_outs, hw_shape)
        psp_outs = self.psp_norm(psp_outs)
        psp_outs = psp_outs.view(-1, *hw_shape, psp_outs.shape[2]).permute(0, 3, 1, 2).contiguous()
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)

        hw_shape = []

        for i in range(len(inputs) - 1):
            hw_shape.append(inputs[i].shape[2:])
            inputs[i] = inputs[i].flatten(2).transpose(1, 2)

        # build laterals
        laterals_swin = [
            lateral_swin(inputs[i], hw_shape[i])
            for i, lateral_swin in enumerate(self.lateral_swins)
        ]

        lateral_norm = [
            lateral_swin(laterals_swin[i])
            for i, lateral_swin in enumerate(self.lateral_norms)
        ]

        for i in range(len(lateral_norm)):
            lateral_norm[i] = lateral_norm[i].view(-1, *hw_shape[i], lateral_norm[i].shape[2]).permute(0, 3, 1, 2).contiguous()

        laterals = [
            laterals_conv(lateral_norm[i])
            for i, laterals_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
