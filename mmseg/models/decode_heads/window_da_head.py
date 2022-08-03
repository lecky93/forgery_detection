# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn

from mmseg.core import add_prefix
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead

from mmseg.ops import SwinBlockSequence



class Swin_PAM(SwinBlockSequence):
    """Window Position Attention Module (PAM)

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels):
        super(Swin_PAM, self).__init__(
                embed_dims=in_channels,
                depth=18,
                num_heads=16,
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

        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""

        hw_shape = (x.shape[2:])
        x_flatten = x.flatten(2).transpose(1, 2)
        out = super(Swin_PAM, self).forward(x_flatten, hw_shape)
        out = out.view(-1, *hw_shape, out.shape[2]).permute(0, 3, 1, 2).contiguous()
        out = self.gamma(out) + x
        return out


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
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
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out


@HEADS.register_module()
class Window_DAHead(BaseDecodeHead):
    """Dual Attention Network for Scene Segmentation.

    This head is the implementation of `DANet
    <https://arxiv.org/abs/1809.02983>`_.

    Args:
        pam_channels (int): The channels of Position Attention Module(PAM).
    """

    def __init__(self, **kwargs):
        super(Window_DAHead, self).__init__(**kwargs)
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam = Swin_PAM(self.channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam = CAM()
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
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
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        feat_sum = pam_feat + cam_feat
        pam_cam_out = self.cls_seg(feat_sum)

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
                super(Window_DAHead, self).losses(pam_cam_seg_logit, seg_label),
                'pam_cam'))
        loss.update(
            add_prefix(
                super(Window_DAHead, self).losses(pam_seg_logit, seg_label), 'pam'))
        loss.update(
            add_prefix(
                super(Window_DAHead, self).losses(cam_seg_logit, seg_label), 'cam'))
        return loss