# model settings
model = dict(
    type='RetinaNet',
    pretrained='http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
    backbone=dict(
        type='EfficientNet',
        model_name='efficientnet-b3',
        out_indices=(4, 7, 17, 25)),
    neck=dict(
        type='FPN',
        #in_channels=[256, 512, 1024, 2048],
        # in_channels=[32, 48, 136, 384],
        in_channels=[24, 40, 112, 320],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
dataset_type = 'BDDDataset'
data_root = '/home/member/Workspace/tuan/dataset/bdd100k_voc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
        # img_scale=(1333, 800),
        ann_file=data_root + 'train/train.txt',
        img_prefix=data_root + 'train/',
        # img_scale=(1280, 640),
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True, 
        resize_keep_ratio=False),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        # img_scale=(1333, 800),
        ann_file=data_root + 'val/val.txt',
        img_prefix=data_root + 'val/',
        # img_scale=(1280, 640),
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        # img_scale=(1333, 800),
        ann_file=data_root + 'val/val.txt',
        img_prefix=data_root + 'val/',
        # img_scale=(1280, 640),
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
#optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#lr_config = dict(
#     policy='cosine',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0/3,
#     target_lr=0.0001,
#     by_epoch=False)
#
checkpoint_config = dict(interval=4)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_efficientnet_b3_fpn_1x_bdd_no_upsample_squeeze'
load_from = None
resume_from =None
# resume_from ='./work_dirs/retinanet_efficientnet_b3_fpn_1x_bdd_no_upsample_squeeze/latest.pth'
workflow = [('train', 1)]

# add pruning config 
prun = dict(
   rate_norm=1,
   rate_dist=1,
   layer_begin=0,
   layer_end=194,
   layer_inter=3,
   skip_downsample=1,
   use_cuda=True)
