import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class WindowNonLocalDenoising(nn.Module):
    """
    Block-wise (windowed) Non-Local Means for BEV feature denoising.
    Applies self-similarity-based attention within local windows to reduce noise.
    """

    def __init__(self,
                 in_channels=256,
                 window_size=15,
                 embed=True,
                 softmax=True,
                 zero_init=True,
                 residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.embed = embed
        self.softmax = softmax
        self.zero_init = zero_init
        self.residual = residual

        if self.embed:
            c_embed = 144 # max(in_channels // 2, 1)
            self.theta_conv = nn.Conv2d(in_channels, c_embed, kernel_size=1, bias=False)
            self.phi_conv = nn.Conv2d(in_channels, c_embed, kernel_size=1, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.theta_conv.weight, std=0.01)
                nn.init.normal_(self.phi_conv.weight, std=0.01)
        else:
            self.theta_conv = self.phi_conv = nn.Identity()

        self.projection = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        if self.zero_init:
            nn.init.zeros_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        # 自動補齊 padding 使其可被 window size 整除
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # [B, C, H_pad, W_pad]
        Hp, Wp = x.shape[2:]

        # Unfold into windows: [B, C, Nh, Nw, ws, ws] → [B*Nh*Nw, C, ws, ws]
        x_patches = x.unfold(2, ws, ws).unfold(3, ws, ws)
        Nh, Nw = x_patches.shape[2:4]
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_patches = x_patches.view(-1, C, ws, ws)  # [B×Nh×Nw, C, ws, ws]

        # Denoising
        y_patches = self._block_nlm(x_patches)  # [B×Nh×Nw, C, ws, ws]

        # Fold back: [B×Nh×Nw, C, ws, ws] → [B, C, Hp, Wp]
        y = y_patches.view(B, Nh, Nw, C, ws, ws)
        y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)

        # Remove padding
        y = y[:, :, :H, :W]

        return x[:, :, :H, :W] + y if self.residual else y

    def _block_nlm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_p, C, ws, ws]
        theta = self.theta_conv(x)
        phi = self.phi_conv(x)
        g = x

        B_p, C_emb, ws, _ = theta.shape
        window_area = ws * ws

        theta = theta.view(B_p, C_emb, -1).transpose(1, 2)  # [B_p, N, C_emb]
        phi = phi.view(B_p, C_emb, -1)                      # [B_p, C_emb, N]
        g = g.view(B_p, self.in_channels, -1).transpose(1, 2)  # [B_p, N, C]

        # Similarity: [B_p, N, N]
        affinity = torch.bmm(theta, phi)
        if self.softmax:
            affinity = affinity / (C_emb ** 0.5)
            affinity = F.softmax(affinity, dim=-1)

        # Weighted feature
        out = torch.bmm(affinity, g)  # [B_p, N, C]
        out = out.transpose(1, 2).contiguous().view(B_p, self.in_channels, ws, ws)
        return self.projection(out)
