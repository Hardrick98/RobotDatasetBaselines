_base_ = ['mmpose::_base_/default_runtime.py']

custom_imports = dict(imports=['rtmpose3d'], allow_failed_imports=False)

# Visualizer
visualizer = dict(
    type='Pose3dLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

# Runtime
max_epochs = 30
base_lr = 5e-4
num_keypoints = 14
randomness = dict(seed=2024)

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

auto_scale_lr = dict(base_batch_size=64)

# Codec settings
codec = dict(
    type='SimCC3DLabel',
    input_size=(288, 384, 288),
    sigma=(6., 6.93, 6.),
    simcc_split_ratio=2.0,
    normalize=True,   
    root_index=0,   
    use_dark=False,
)

#backbone_path = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'
backbone_path = "/work/ToyotaHPE/rcatalini/EventRobotPose/mmpose/baselines/RTMPose/epoch_9.pth"

# model settings
model = dict(
    type='TopdownPoseEstimator3D',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=backbone_path)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMW3DHead',
        in_channels=1024,
        out_channels=14,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=[
            dict(
                type='KLDiscretLossWithWeight',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            dict(
                type='BoneLoss',
                joint_parents=[0, 0, 0, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12],
                use_target_weight=True,
                loss_weight=2.0)
        ],
        decoder=codec),
    # test_cfg=dict(flip_test=False, mode='2d')
    test_cfg=dict(flip_test=False))

# base dataset settings
data_mode = 'topdown'
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(288, 384)),
    #dict(type='YOLOXHSVRandomAug'),
    #dict(type='Albumentation', transforms=[
    #    dict(type='Blur', p=0.1),
    #    dict(type='MedianBlur', p=0.1),
        #dict(type='CoarseDropout', max_holes=1, max_height=0.4, max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, p=1.0)
    #]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]


val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RobotDataset3D',
        ann_file='train_coco_pose3d_sampled.json',
        data_root='/work/ToyotaHPE/rcatalini/EventRobotPose/exo_dataset/',
        seq_len=1,
        pipeline=train_pipeline,
        test_mode=False
    )
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='RobotDataset3D',
        ann_file='val_coco_pose3d_sampled.json',
        data_root='/work/ToyotaHPE/rcatalini/EventRobotPose/exo_dataset/',
        seq_len=1,
        test_mode=True,
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='RobotDataset3D',
        ann_file='test_coco_pose3d.json',
        data_root='/work/ToyotaHPE/rcatalini/EventRobotPose/exo_dataset/',
        seq_len=1,
        test_mode=True,
        pipeline=val_pipeline
    )
)

#custom_hooks = [
#    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49),
#    dict(type='mmdet.PipelineSwitchHook', switch_epoch=max_epochs - stage2_num_epochs, switch_pipeline=train_pipeline_stage2)
#]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='MPJPE', rule='less', max_keep_ckpts=10)
)

#val_evaluator = [dict(type='SimpleMPJPE', mode='mpjpe'), dict(type='SimpleMPJPE', mode='p-mpjpe')]
#test_evaluator = val_evaluator

val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator