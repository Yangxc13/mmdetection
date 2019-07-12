# model settings
model = dict(
    type='FOVEA',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1, # C3, C4, C5
        num_outs=5, # P3, P4, P5, P6, P7
        add_extra_convs=True, # for generate P6,P7, conv or max_pool2d
        extra_convs_on_inputs=True, # C5->P6 or P5->P6
        relu_before_extra_convs=True), # P6->ReLU->P7 or P6->P7
    bbox_head=dict(
        type='FoveaHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4, # 与neck的num_outs=5无关，neck的tuple len=5的输出，通过multi_apply分别通过这个4层的神经网络，
        feat_channels=256,
        strides=[8, 16, 32, 64, 128], # 2^l, l=3,4,...7
        base_edge_list=[32, 64, 126, 256, 512],
        scale_ranges=((16,64), (32,128), (64,256), (128,512), (256,1024)), # base_area: (2^(l+2))^2 allowed_area: (2^(l+1))^2~(2^(l+3))^2
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # TODO 是否考虑GETTING_STARTED.md中的Linear Scaling Rule?
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
gpus=1
# device_ids = range(3) 在本版本代码中没有地方使用了这个
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/foveanet_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
