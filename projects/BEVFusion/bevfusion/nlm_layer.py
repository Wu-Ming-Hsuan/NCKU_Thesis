import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class WindowNonLocalDenoising(nn.Module):
    """
    Block-wise (window) Non-Local Means for BEV feature denoising.
    Avoids OOM by restricting affinity to each patch (window).
    """
    def __init__(self, in_channels, window_size=18, embed=True, softmax=True, zero_init=True):
        super().__init__()
        self.in_channels = in_channels
        self.embed = embed
        self.softmax = softmax
        self.zero_init = zero_init
        self.window_size = window_size

        # 只建立一次
        if self.embed:
            c_embed = max(in_channels // 2, 1)
            self.theta_conv = nn.Conv2d(in_channels, c_embed, 1, bias=False)
            self.phi_conv   = nn.Conv2d(in_channels, c_embed, 1, bias=False)
            nn.init.normal_(self.theta_conv.weight, std=0.01)
            nn.init.normal_(self.phi_conv.weight, std=0.01)
        else:
            self.theta_conv = self.phi_conv = nn.Identity()

        self.projection = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        if self.zero_init:
            nn.init.zeros_(self.projection.weight)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H/W必須能整除window size"

        # 切 patch [B, C, Nh, ws, Nw, ws] -> [B*Nh*Nw, C, ws, ws]
        x_patches = x.unfold(2, ws, ws).unfold(3, ws, ws)  # [B, C, Nh, Nw, ws, ws]
        Nh, Nw = x_patches.shape[2:4]
        x_patches = x_patches.permute(0,2,3,1,4,5).contiguous().view(-1, C, ws, ws) # [B*Nh*Nw, C, ws, ws]

        # NLM Denoising on every patch
        y_patches = self._block_nlm(x_patches)  # [B*Nh*Nw, C, ws, ws]

        # 拼回 [B, C, H, W]
        y = y_patches.view(B, Nh, Nw, C, ws, ws).permute(0,3,1,4,2,5).contiguous()
        y = y.view(B, C, H, W)
        return x + y  # 殘差結構

    def _block_nlm(self, x):
        # x: [B_p, C, ws, ws]
        theta = self.theta_conv(x)      # [B_p, C_emb, ws, ws]
        phi = self.phi_conv(x)
        g = x                          # [B_p, C, ws, ws]
        Bp, C_emb, ws, _ = theta.shape
        theta = theta.view(Bp, C_emb, -1).transpose(1,2)   # [B_p, ws*ws, C_emb]
        phi   = phi.view(Bp, C_emb, -1)                    # [B_p, C_emb, ws*ws]
        g     = g.view(Bp, g.shape[1], -1).transpose(1,2)  # [B_p, ws*ws, C]

        f = torch.bmm(theta, phi)      # [B_p, ws*ws, ws*ws]
        if self.softmax:
            f = f / (C_emb ** 0.5)
            f = F.softmax(f, dim=-1)
        y = torch.bmm(f, g)            # [B_p, ws*ws, C]
        y = y.transpose(1,2).contiguous().view(Bp, g.shape[2], ws, ws) # [B_p, C, ws, ws]
        y = self.projection(y)
        return y