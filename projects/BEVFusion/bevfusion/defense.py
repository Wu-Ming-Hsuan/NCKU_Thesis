import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from mmdet3d.registry import MODELS

################################################################################
# PNLM — Patch-wise Non-Local Mean denoising block
################################################################################

@MODELS.register_module()
class PNLM(nn.Module):
    @staticmethod
    def _pad(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, int, int]:
        """Reflect‑pad so H,W are multiples of *k*; return (x_pad, ph, pw)."""
        _, _, h, w = x.shape
        ph, pw = (k - h % k) % k, (k - w % k) % k
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), ph, pw

    @staticmethod
    def _unpad(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        if ph:
            x = x[..., :-ph, :]
        if pw:
            x = x[..., :-pw]
        return x

    @staticmethod
    def _blockify(x: torch.Tensor, k: int) -> torch.Tensor:
        """Split B×C×H×W into blocks (B',C,k,k) where B'=B·H//k·W//k."""
        b, c, h, w = x.shape
        x = x.view(b, c, h // k, k, w // k, k)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(-1, c, k, k)

    @staticmethod
    def _unblockify(blocks: torch.Tensor, b: int, c: int, h: int, w: int, k: int) -> torch.Tensor:
        nh, nw = h // k, w // k
        x = blocks.view(b, nh, nw, c, k, k).permute(0, 3, 1, 4, 2, 5)
        return x.contiguous().view(b, c, h, w)

    # ------------------------------------------------------------------ ctor
    def __init__(
        self,
        in_channels: int = 256,
        window_size: int = 15,
        embed_channels: int = 144,
        use_softmax: bool = True,
        zero_init_proj: bool = True,
        with_residual: bool = True,
    ) -> None:
        super().__init__()
        self.c = in_channels
        self.k = window_size
        self.use_softmax = use_softmax
        self.with_residual = with_residual

        c_emb = embed_channels #max(in_channels // 2, 1)
        self.to_q = nn.Conv2d(in_channels, c_emb, 1, bias=False) if c_emb != in_channels else nn.Identity()
        self.to_k = nn.Conv2d(in_channels, c_emb, 1, bias=False) if c_emb != in_channels else nn.Identity()
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        if zero_init_proj:
            nn.init.zeros_(self.proj_out.weight)

    def _attention(self, blocks: torch.Tensor) -> torch.Tensor:
        b, _, k, _ = blocks.shape
        q = self.to_q(blocks).flatten(2).transpose(1, 2)  # (B',N,Ce)
        k_mat = self.to_k(blocks).flatten(2)              # (B',Ce,N)
        v = blocks.flatten(2).transpose(1, 2)             # (B',N,C)
        att = q @ k_mat                                   # (B',N,N)
        if self.use_softmax:
            att = F.softmax(att / math.sqrt(q.size(-1)), dim=-1)
        y = (att @ v).transpose(1, 2).view(b, self.c, k, k)
        return self.proj_out(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_pad, ph, pw = self._pad(x, self.k)
        blocks = self._blockify(x_pad, self.k)
        y_blocks = self._attention(blocks)
        y = self._unblockify(y_blocks, b, c, *x_pad.shape[2:], self.k)
        y = self._unpad(y, ph, pw)
        return x + y if self.with_residual else y


################################################################################
# AMHSA — Augmented multi‑head self-attention denoising block
################################################################################

@MODELS.register_module()
class AMHSA(nn.Module):
    """Symmetric Multi‑Scale **multi‑head** self‑attention with residual.

    Method order：helpers → __init__ → _attention → forward
    """

    # -------------------------- helper functions (static) -------------------
    @staticmethod
    def _aug(x: torch.Tensor) -> List[torch.Tensor]:
        """rotat: 0°, 90°, 180°, 270° + H-flip + V-flip."""
        return [
            x,
            torch.flip(x, (-1,)),
            torch.flip(x, (-2,)),
            torch.rot90(x, 1, (-2, -1)),
            torch.rot90(x, 2, (-2, -1)),
            torch.rot90(x, 3, (-2, -1)),
        ]

    # ----------------------- multi‑scale symmetric ---------------------
    def _ms(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        rot_z = self._aug(z)
        feats: List[torch.Tensor] = []
        for conv in self.dilated:
            y = conv(z)
            feats.extend(y * r for r in rot_z)
        return self.fuse(torch.cat(feats, 1))

    def __init__(
        self,
        in_channels: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        patch: int = 1,
        reduction: int = 18,
        dilations: Tuple[int, ...] = (1, 2, 3),
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.scale_qk = 1 / math.sqrt(self.dh)
        self.res_scale = residual_scale

        # patch embed / de‑embed
        self.to_tokens = nn.Conv2d(in_channels, d_model, patch, patch)
        self.to_feats = nn.ConvTranspose2d(d_model, in_channels, patch, patch)

        # multi‑scale symmetric bias branch
        c_mid = d_model // reduction
        self.reduce = nn.Conv2d(d_model, c_mid, 1)
        self.dilated = nn.ModuleList([
            nn.Conv2d(c_mid, c_mid, 3, padding=d, dilation=d) for d in dilations
        ])
        self.fuse = nn.Conv2d(c_mid * len(dilations) * 6, d_model, 1)

        # QKV & output projections
        self.qkv = nn.Conv2d(d_model, d_model * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(d_model, d_model, 1, bias=False)
        nn.init.zeros_(self.proj_out.weight)

    # -------------- 🧩 multi‑head scaled‑dot‑product attention --------------
    def _attention(self, token: torch.Tensor) -> torch.Tensor:
        """Apply multi‑head attention over flattened spatial tokens."""
        b, _, h_t, w_t = token.shape
        n = h_t * w_t
        qkv = self.qkv(token).reshape(b, 3, self.nh, self.dh, n)
        q, k, v = qkv.unbind(1)  # each: (B, nh, dh, N)

        att = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale_qk
        att = att.softmax(-1)
        out = torch.einsum("bhnm,bhdm->bhdn", att, v)
        return out.reshape(b, -1, h_t, w_t)  # merge heads

    # ------------------------------ forward pass ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full pipeline: embed → add bias → attention → project → residual."""
        # 1) Patch‑to‑token embed
        token = self.to_tokens(x)

        # 2) Add symmetric multi‑scale bias
        token = token + self._ms_bias(token)

        # 3) Multi‑head attention in token space
        att_out = self._attention(token)
        att_out = self.proj_out(att_out)
        token = token + att_out  # residual in token space

        # 4) Token‑to‑feature and residual to input
        feat = self.to_feats(token)
        return x + feat * self.res_scale
    
# --------------------------------------------------------------------------- #
# 共用：高斯 kernel 產生器
# --------------------------------------------------------------------------- #
def _gaussian_kernel(ksize: int, sigma: float, dtype=torch.float32):
    ax = torch.arange(ksize, dtype=dtype) - ksize // 2
    gauss = torch.exp(-0.5 * (ax ** 2) / sigma ** 2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d

# --------------------------------------------------------------------------- #
# 1. GaussianBlur (feature)
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class GaussianBlur(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        ksize: int = 5,
        sigma: float = 1.5,
        with_residual: bool = True,
        res_scale: float = 0.2,
    ):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 0
        kernel = _gaussian_kernel(ksize, sigma)   # (k,k)
        self.register_buffer(
            "weight",
            kernel.expand(in_channels, 1, ksize, ksize)
        )
        self.groups = in_channels
        self.pad = ksize // 2
        self.with_residual = with_residual
        self.res_scale = res_scale

    def forward(self, x):
        y = F.conv2d(x, self.weight, padding=self.pad, groups=self.groups)
        return x + self.res_scale * (y - x) if self.with_residual else y

# --------------------------------------------------------------------------- #
# 2. MedianBlur
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class MedianBlur(nn.Module):
    def __init__(
        self,
        ksize: int = 3,
        with_residual: bool = True,
        res_scale: float = 0.2,
    ):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 1
        self.ksize = ksize
        self.with_residual = with_residual
        self.res_scale = res_scale

    def forward(self, x):
        B, C, H, W = x.shape
        unfold = F.unfold(x, self.ksize, padding=self.ksize // 2)  # (B, C·k², H·W)
        med = unfold.view(B, C, self.ksize ** 2, H, W).median(dim=2).values
        return x + self.res_scale * (med - x) if self.with_residual else med

# --------------------------------------------------------------------------- #
# 3. MeanBlur (= AvgPool)
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class MeanBlur(nn.Module):
    def __init__(
        self,
        ksize: int = 3,
        with_residual: bool = True,
        res_scale: float = 0.2,
    ):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 1
        self.ksize = ksize
        self.with_residual = with_residual
        self.res_scale = res_scale

    def forward(self, x):
        y = F.avg_pool2d(x, self.ksize, stride=1, padding=self.ksize // 2)
        return x + self.res_scale * (y - x) if self.with_residual else y

# --------------------------------------------------------------------------- #
# 4. Bit-Depth Reduction (float → n bits quantisation)
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class BitDepth(nn.Module):
    def __init__(
        self,
        bits: int = 5,
        with_residual: bool = False,  # 通常直接量化不殘差
    ):
        super().__init__()
        assert 1 <= bits <= 16
        self.levels = 2 ** bits
        self.with_residual = with_residual

    def forward(self, x):
        # 假設 feature 已經經過 BN，大多分佈在 [-3,3]；用 tanh 壓到 [-1,1] 再做均勻量化
        z = torch.tanh(x)
        y = torch.round((z + 1) * (self.levels / 2 - 1)) / (self.levels / 2 - 1) - 1
        # 反投回原空間
        y = 0.5 * torch.log((1 + y) / (1 - y) + 1e-6)
        return x + (y - x) if self.with_residual else y

# --------------------------------------------------------------------------- #
# 5. Bilateral-like (spatial Gaussian × channel-wise BN weight)
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class Bilateral(nn.Module):
    """簡化 Bilateral：空間高斯 + 通道標準差當作 range term 權重。"""

    def __init__(
        self,
        in_channels: int = 256,
        ksize: int = 5,
        sigma: float = 1.0,
        with_residual: bool = True,
        res_scale: float = 0.2,
    ):
        super().__init__()
        kernel = _gaussian_kernel(ksize, sigma)
        self.register_buffer("spa", kernel[None, None])  # (1,1,k,k)
        self.pad = ksize // 2
        self.with_residual = with_residual
        self.res_scale = res_scale

    def forward(self, x):
        # 以 channel-wise variance approximate "range" term
        var = x.var(dim=(2, 3), keepdim=True) + 1e-6         # (B,C,1,1)
        weight = self.spa / (1 + var)                        # (B,C,k,k) broadcasting
        y = F.conv2d(x, weight, padding=self.pad, groups=x.size(1))
        return x + self.res_scale * (y - x) if self.with_residual else y

# --------------------------------------------------------------------------- #
# 6. Non-Local Means (NLMean-Lite)
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class NLMFeat(nn.Module):
    """局部 NLM：在 r×r 視窗上計算相似度後加權平均（簡化版）."""

    def __init__(
        self,
        search_radius: int = 3,
        h: float = 0.1,
        with_residual: bool = True,
        res_scale: float = 0.2,
    ):
        super().__init__()
        self.r = search_radius
        self.h2 = h * h
        self.with_residual = with_residual
        self.res_scale = res_scale

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.r
        unfold = F.unfold(x, 2 * pad + 1, padding=pad)   # (B, C·k², H·W)
        neigh = unfold.view(B, C, -1, H, W)              # (B,C,k²,H,W)
        center = x.unsqueeze(2)                          # (B,C,1,H,W)
        dist2 = (neigh - center).pow(2).mean(1, keepdim=True)  # (B,1,k²,H,W)
        weight = torch.exp(-dist2 / self.h2)             # Gaussian in feature space
        weight = weight / weight.sum(2, keepdim=True)
        y = (weight * neigh).sum(2)                      # (B,C,H,W)
        return x + self.res_scale * (y - x) if self.with_residual else y
