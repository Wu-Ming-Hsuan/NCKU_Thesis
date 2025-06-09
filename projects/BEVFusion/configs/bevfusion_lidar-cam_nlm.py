_base_ = [
    './bevfusion_lidar-cam.py'
]

# Load pretrained checkpoint
load_from = "checkpoints/BEVFusion_cam_lidar_nlm.pth"

# Add NLM layer configuration
model = dict(
    nlm_layer=dict(
        type='WindowNonLocalDenoising',
        in_channels=256,
        embed=True,
        softmax=True,
        zero_init=True
    ),
    freeze_except=['nlm_layer']  # 凍結除了 nlm_layer 以外的所有模塊
)

custom_hooks = [
    dict(
        type='AttackHook',
        attack_cfg=dict(
            type='AutoPGD'
        )
    )
]
# 使用較小的 batch size 以適應訓練
train_dataloader = dict(batch_size=1)
