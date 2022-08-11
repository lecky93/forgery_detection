# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn

from mmseg.core import add_prefix
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmseg.ops import SwinBlockSequence
from mmseg.ops import resize
from mmcv.cnn import build_norm_layer


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class Swin_PAM(SwinBlockSequence):
    """Window Position Attention Module (PAM)

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels, depth, num_head):
        super(Swin_PAM, self).__init__(
                embed_dims=in_channels,
                depth=depth,
                num_heads=num_head,
                feedforward_channels=int(4 * in_channels),
                window_size=7,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                upsample=None,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                with_cp=False,
                init_cfg=None)

        self.norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        hw_shape = (x.shape[2:])
        x_flatten = x.flatten(2).transpose(1, 2)
        out = super(Swin_PAM, self).forward(x_flatten, hw_shape)
        out = self.norm(out)
        out = out.view(-1, *hw_shape, out.shape[2]).permute(0, 3, 1, 2).contiguous()
        out = self.gamma(out)
        return out

class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self, in_channels):
        super(CAM, self).__init__()

        self.norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma(out)
        return out

class DABlock(nn.Module):
    def __init__(self, in_channels, channels, depth, num_head, conv_cfg, norm_cfg):
        super(DABlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.depth = depth
        self.num_head = num_head

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv_in = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='GELU'),
            inplace=False
        )
        if self.depth != 0:
            # self.constrained_conv = BayarConv2d(in_channels=channels, out_channels=channels, padding=2)
            self.pam = Swin_PAM(self.channels, self.depth, self.num_head)
            self.cam = CAM(self.channels)

            # self.conv_out = ConvModule(
            #     self.channels * 2,
            #     self.channels,
            #     3,
            #     padding=1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=dict(type='GELU'),
            #     inplace=False
            # )

    def forward(self, x):
        x = self.conv_in(x)
        if self.depth == 0:
            return x, x, x
        # constrain = self.constrained_conv(x)
        pam_out = x + self.pam(x)
        cam_out = x + self.cam(x)
        # pam_cam_out = pam_out + cam_out
        pam_cam_out = torch.cat([pam_out, cam_out], dim=1)
        pam_cam_out = self.conv_out(pam_cam_out)
        return pam_cam_out, pam_out, cam_out

@HEADS.register_module()
class Swin_DAHead2(BaseDecodeHead):
    """Dual Attention Network for Scene Segmentation.

    This head is the implementation of `DANet
    <https://arxiv.org/abs/1809.02983>`_.

    Args:
        pam_channels (int): The channels of Position Attention Module(PAM).
    """

    def __init__(self, depths=[0, 2, 18, 2], num_heads=[4, 8, 16, 32], **kwargs):
        super(Swin_DAHead2, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.depths = depths
        self.num_heads = num_heads

        self.lateral_da = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i, in_channels in enumerate(self.in_channels):
            da_block = DABlock(in_channels, self.channels, depths[i], num_heads[i], self.conv_cfg, self.norm_cfg)
            self.lateral_da.append(da_block)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='GELU'),
                inplace=False)
            self.fpn_convs.append(fpn_conv)


        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='GELU'))

        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.pam_conv_seg(feat)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cam_conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        pam_laterals = []
        cam_laterals = []
        pam_cam_laterals = []
        for i, lateral_da in enumerate(self.lateral_da):
            pam_cam_lat, pam_lat, cam_lat = lateral_da(x[i])
            pam_laterals.append(pam_lat)
            cam_laterals.append(cam_lat)
            pam_cam_laterals.append(pam_cam_lat)

        # build top-down path
        used_backbone_levels = len(x)

        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = pam_cam_laterals[i - 1].shape[2:]

            pam_laterals[i - 1] = pam_laterals[i - 1] + resize(
                pam_laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

            cam_laterals[i - 1] = cam_laterals[i - 1] + resize(
                cam_laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

            pam_cam_laterals[i - 1] = pam_cam_laterals[i - 1] + resize(
                pam_cam_laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](pam_cam_laterals[i])
            for i in range(used_backbone_levels)
        ]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        pam_out = self.pam_cls_seg(pam_laterals[0])
        cam_out = self.cam_cls_seg(cam_laterals[0])
        pam_cam_out = self.cls_seg(feats)

        return pam_cam_out, pam_out, cam_out

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        pam_cam_seg_logit, pam_seg_logit, cam_seg_logit = seg_logit
        loss = dict()
        loss.update(
            add_prefix(
                super(Swin_DAHead2, self).losses(pam_cam_seg_logit, seg_label),
                'pam_cam'))
        loss.update(
            add_prefix(
                super(Swin_DAHead2, self).losses(pam_seg_logit, seg_label), 'pam'))
        loss.update(
            add_prefix(
                super(Swin_DAHead2, self).losses(cam_seg_logit, seg_label), 'cam'))
        return loss