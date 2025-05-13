_base_ = [
    './bevfusion_lidar-cam.py'
]

# Load pretrained checkpoint
load_from = "/ssddd/matthew/NCKU_Thesis/checkpoints/BEVFusion_cam_lidar.pth"

# Add NLM layer configuration
model = dict(
    nlm_layer=dict(
        type='NonLocalDenoising',
        embed=True,
        softmax=True,
        zero_init=True
    ),
    _freeze_except=['nlm_layer']  # 凍結除了 nlm_layer 以外的所有模塊
)

# 只對 nlm_layer 使用較高的學習率，其他層凍結
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0),
            'img_neck': dict(lr_mult=0),
            'pts_middle_encoder': dict(lr_mult=0),
            'pts_backbone': dict(lr_mult=0),
            'pts_neck': dict(lr_mult=0),
            'bbox_head': dict(lr_mult=0),
            'nlm_layer': dict(lr_mult=1.0)
        }
    )
)

# 使用較小的 batch size 以適應訓練
train_dataloader = dict(batch_size=1)
