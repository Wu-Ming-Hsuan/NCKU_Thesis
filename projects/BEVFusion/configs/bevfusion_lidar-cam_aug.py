_base_ = [
    './bevfusion_lidar-cam.py'
]

# Load pretrained checkpoint
load_from = "checkpoints/BEVFusion_cam_lidar.pth"

# Add NLM layer configuration
model = dict(
    fractal_enhancer=dict(
        type='FractalEnhancer',
    ),
    freeze_except=['fractal_enhancer']  # 凍結除了 nlm_layer 以外的所有模塊
)
# 使用較小的 batch size 以適應訓練
train_dataloader = dict(batch_size=1)
