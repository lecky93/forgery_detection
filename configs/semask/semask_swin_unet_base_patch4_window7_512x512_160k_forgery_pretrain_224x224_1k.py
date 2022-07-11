_base_ = [
    '../_base_/datasets/forgery.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

img_size = (512, 512)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/swin_base_patch4_window7_224.pth',
    backbone=dict(
        type='SeMaskSwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        num_cls=2,
        sem_window_size=7,
        num_sem_blocks=1,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1),
    decode_head=dict(
        type='SwinTransformerUnetHead',
        img_size=512,
        patch_size=4,
        num_classes=2,
        embed_dim=96,
        depths_decoder=[2, 18, 2, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        for_se_mask=True,
        in_index=['res2', 'res3', 'res4', 'res5'],
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[1, 10]),
            dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=1.0, per_image=True, class_weight=[1, 10])]),
    auxiliary_head=dict(
        type='SeMaskSemanticHead',
        in_channels=[2, 2, 2, 2],
        img_size=img_size,
        in_index=['res2', 'res3', 'res4', 'res5'],
        feature_strides=[4, 8, 16, 32],
        channels=2,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1, 10])),
    train_cfg=dict(),
    test_cfg=dict(mode='sliding'))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)

gpu_ids = [0]
auto_resume = False
