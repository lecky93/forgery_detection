# dataset settings
dataset_type = 'CasiaDataset'
data_root = '../dataset/forgery'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_casia_train = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='casia/CASIA2/image',
    ann_dir='casia/CASIA2/ann',
    pipeline=train_pipeline
)

dataset_defacto_copy_move_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/copy-move/copymove_img/img',
    ann_dir='defacto/copy-move/copymove_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_inpainting_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/inpainting/inpainting_img/img',
    ann_dir='defacto/inpainting/inpainting_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice1_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_1_img/img',
    ann_dir='defacto/splice/splicing_1_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice2_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_2_img/img',
    ann_dir='defacto/splice/splicing_2_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice3_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_3_img/img',
    ann_dir='defacto/splice/splicing_3_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice4_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_4_img/img',
    ann_dir='defacto/splice/splicing_4_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice5_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_5_img/img',
    ann_dir='defacto/splice/splicing_5_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice6_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_6_img/img',
    ann_dir='defacto/splice/splicing_6_annotations/probe_ann',
    pipeline=train_pipeline
)

dataset_defacto_splice7_train = dict(
    type='DefactoDataset',
    data_root=data_root,
    img_dir='defacto/splice/splicing_7_img/img',
    ann_dir='defacto/splice/splicing_7_annotations/probe_ann',
    pipeline=train_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dataset_casia_train,
        dataset_defacto_copy_move_train,
        dataset_defacto_inpainting_train,
        dataset_defacto_splice1_train,
        dataset_defacto_splice2_train,
        dataset_defacto_splice3_train,
        dataset_defacto_splice4_train,
        dataset_defacto_splice5_train,
        dataset_defacto_splice6_train,
        dataset_defacto_splice7_train,
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='casia/CASIA1/image',
        ann_dir='casia/CASIA1/ann',
        pipeline=test_pipeline),
    test=dict(
        type='DefactoDataset',
        data_root=data_root,
        img_dir='defacto/splice/splicing_1_img/img',
        ann_dir='defacto/splice/splicing_1_annotations/probe_ann',
        pipeline=test_pipeline))