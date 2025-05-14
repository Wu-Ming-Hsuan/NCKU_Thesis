_base_ = [
    './bevfusion_lidar-cam.py'
]

# Load pretrained checkpoint
load_from = "/raid/matthew/NCKU_Thesis/checkpoints/BEVFusion_cam_lidar.pth"

# Add NLM layer configuration
model = dict(
    nlm_layer=dict(
        type='NonLocalDenoising',
        embed=True,
        softmax=True,
        zero_init=True
    ),
    freeze_except=['nlm_layer']  # 凍結除了 nlm_layer 以外的所有模塊
)
# 使用較小的 batch size 以適應訓練
train_dataloader = dict(batch_size=1)
