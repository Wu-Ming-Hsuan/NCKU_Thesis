import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from mmdet3d.registry import MODELS

torch.set_float32_matmul_precision('high')

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
        patch: int = 15,
        embed_channels: int = 144,
        use_softmax: bool = True,
        zero_init_proj: bool = True,
        with_residual: bool = True,
    ) -> None:
        super().__init__()
        self.c = in_channels
        self.k = patch
        self.use_softmax = use_softmax
        self.with_residual = with_residual

        c_emb = embed_channels #max(in_channels // 2, 1)
        self.to_q = nn.Conv2d(in_channels, c_emb, 1, bias=False) if c_emb != in_channels else nn.Identity()
        self.to_k = nn.Conv2d(in_channels, c_emb, 1, bias=False) if c_emb != in_channels else nn.Identity()
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        if zero_init_proj:
            nn.init.zeros_(self.proj_out.weight)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attention(self, blocks: torch.Tensor) -> torch.Tensor:
        b, c, k, _ = blocks.shape
        q = self.to_q(blocks).flatten(2).transpose(1, 2).contiguous()
        k_mat = self.to_k(blocks).flatten(2).transpose(1, 2).contiguous()
        v = blocks.flatten(2).transpose(1, 2).contiguous()
        
        y = F.scaled_dot_product_attention(q, k_mat, v)
        y = y.transpose(1, 2).view(b, c, k, k).contiguous()
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
        patch: int = 2,
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------- 🧩 multi‑head scaled‑dot‑product attention --------------
    def _attention(self, token: torch.Tensor) -> torch.Tensor:
        b, _, h_t, w_t = token.shape
        qkv = self.qkv(token)
        q, k, v = qkv.reshape(b, 3, self.nh, self.dh, -1).unbind(1)
        q = q.permute(0, 2, 3, 1).contiguous() # B, nh, N, dh
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.permute(0, 2, 3, 1).contiguous()

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 3, 1, 2).reshape(b, -1, h_t, w_t).contiguous() # (B, nh, N, dh) -> (B, C, H, W)
        return out

    # ------------------------------ forward pass ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full pipeline: embed → add bias → attention → project → residual."""
        # 1) Patch‑to‑token embed
        token = self.to_tokens(x)

        # 2) Add symmetric multi‑scale bias
        token = token + self._ms(token)

        # 3) Multi‑head attention in token space
        att_out = self._attention(token)
        att_out = self.proj_out(att_out)
        token = token + att_out  # residual in token space

        # 4) Token‑to‑feature and residual to input
        feat = self.to_feats(token)
        return x + feat * self.res_scale

def _gaussian_kernel(k: int, s: float, dtype=torch.float32):
    ax = torch.arange(k, dtype=dtype) - k // 2
    g = torch.exp(-0.5 * (ax ** 2) / s ** 2)
    ker = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return ker

def _to_np(t: torch.Tensor):
    if t.ndim == 3: t = t.permute(1, 2, 0)
    return (t.detach().cpu().numpy() * 255).astype("uint8")

def _to_tensor(arr, ref: torch.Tensor):
    return torch.from_numpy(arr).permute(2, 0, 1).to(ref.device).float() / 255.0
# =============================================================== #

# 1★ GaussianBlur －－＊
@MODELS.register_module()
class GaussianBlur(nn.Module):
    def __init__(self, in_channels=256, ksize=5, sigma=1.5,
                 with_residual=True, res_scale=0.2):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 0
        ker = _gaussian_kernel(ksize, sigma).expand(in_channels, 1, ksize, ksize)
        self.register_buffer("w", ker)
        self.g, self.p = in_channels, ksize // 2
        self.residual, self.r = with_residual, res_scale

    def forward(self, x):
        y = F.conv2d(x, self.w, padding=self.p, groups=self.g)
        return x + self.r * (y - x) if self.residual else y

# 2★ MedianBlur －－＊
@MODELS.register_module()
class MedianBlur(nn.Module):
    def __init__(self, ksize=3, with_residual=True, res_scale=0.2):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 1
        self.k, self.residual, self.r = ksize, with_residual, res_scale

    def forward(self, x):
        B, C, H, W = x.shape
        u = F.unfold(x, self.k, padding=self.k // 2)          # (B,Ck²,HW)
        med = u.view(B, C, self.k ** 2, H, W).median(2).values
        return x + self.r * (med - x) if self.residual else med

# 3★ MeanBlur (=AvgPool) －－＊
@MODELS.register_module()
class MeanBlur(nn.Module):
    def __init__(self, ksize=3, with_residual=True, res_scale=0.2):
        super().__init__()
        assert ksize % 2 == 1 and ksize > 1
        self.k, self.residual, self.r = ksize, with_residual, res_scale

    def forward(self, x):
        y = F.avg_pool2d(x, self.k, stride=1, padding=self.k // 2)
        return x + self.r * (y - x) if self.residual else y

# 4★ BitDepth (feature-squeezing) －－＊
@MODELS.register_module()
class BitDepth(nn.Module):
    def __init__(self, bits=5):
        super().__init__()
        assert 1 <= bits <= 16
        self.levels = 2 ** bits

    def forward(self, x):
        z = torch.tanh(x)
        y = torch.round((z + 1) * (self.levels/2 - 1)) / (self.levels/2 - 1) - 1
        y = 0.5 * torch.log((1 + y) / (1 - y) + 1e-6)
        return y

# 5★ Bilateral (stat-aware) －－＊
@MODELS.register_module()
class Bilateral(nn.Module):
    def __init__(self, in_channels=256, ksize=5, sigma=1.0,
                 with_residual=True, res_scale=0.2):
        super().__init__()
        ker = _gaussian_kernel(ksize, sigma)[None, None]
        self.register_buffer("spa", ker)
        self.pad, self.residual, self.r = ksize // 2, with_residual, res_scale

    def forward(self, x):
        var = x.var(dim=(2, 3), keepdim=True) + 1e-6
        w = (self.spa / (1 + var)).expand_as(self.spa.repeat(x.size(1), 1, 1, 1))
        y = F.conv2d(x, w, padding=self.pad, groups=x.size(1))
        return x + self.r * (y - x) if self.residual else y

# 6★ NLM (local, lite) －－＊
@MODELS.register_module()
class NLM(nn.Module):
    def __init__(self, search_radius=3, h=0.1, with_residual=True, res_scale=0.2):
        super().__init__()
        self.r, self.h2 = search_radius, h * h
        self.residual, self.ratio = with_residual, res_scale

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.r
        u = F.unfold(x, 2 * pad + 1, padding=pad)             # (B,Ck²,HW)
        neigh = u.view(B, C, -1, H, W)
        dist2 = (neigh - x.unsqueeze(2)).pow(2).mean(1, keepdim=True)
        w = torch.exp(-dist2 / self.h2)
        w = w / w.sum(2, keepdim=True)
        y = (w * neigh).sum(2)
        return x + self.ratio * (y - x) if self.residual else y

# 7★ JPEG compression －－＊
@MODELS.register_module()
class JPEG(nn.Module):
    """通用 JPEG：支援 C=1,3,4,>4。＊無可學權重"""
    def __init__(self, quality=60):
        super().__init__()
        self.q = int(quality)

    @torch.no_grad()
    def _jpeg_gray(self, arr):                 # arr : (H,W) uint8
        _, enc = cv2.imencode(
            ".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, self.q]
        )
        return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)       # (H,W)

    @torch.no_grad()
    def forward(self, x):                      # x: (B,C,H,W) in [0,1]
        outs = []
        for feat in x:                         # loop B
            C, H, W = feat.shape
            if C in (1, 3, 4):                 # 直接壓縮
                np_im = (feat.detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                _, enc = cv2.imencode(".jpg", np_im,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.q])
                dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)    # shape 保持
                if dec.ndim == 2: dec = dec[..., None]           # -> (H,W,1)
                out = torch.from_numpy(dec).permute(2, 0, 1).float() / 255.0
            else:                           # C>4：每通道獨立灰階 JPEG
                chans = []
                for c in feat:              # c : (H,W)
                    arr = (c.detach().cpu().numpy() * 255).astype("uint8")
                    dec = self._jpeg_gray(arr)                   # (H,W)
                    chans.append(torch.from_numpy(dec))
                out = torch.stack(chans).float() / 255.0         # (C,H,W)
            outs.append(out.to(feat.device))
        return torch.stack(outs)

# 8★ RandGaussian (randomised smoothing) －－＊
@MODELS.register_module()
class RandGaussian(nn.Module):
    def __init__(self, in_channels=256, kmin=3, kmax=7,
                 sigma_min=0.5, sigma_max=2.0):
        super().__init__()
        self.C, self.kmin, self.kmax = in_channels, kmin, kmax
        self.smin, self.smax = sigma_min, sigma_max

    def forward(self, x):
        k = (torch.randint(self.kmin, self.kmax + 1, ()).item() | 1)
        s = torch.rand(()) * (self.smax - self.smin) + self.smin
        ker = _gaussian_kernel(k, s).expand(self.C, 1, k, k).to(x.device)
        return F.conv2d(x, ker, padding=k // 2, groups=self.C)

# 9★ SpatialSmooth (uniform kernel) －－＊
@MODELS.register_module()
class SpatialSmooth(nn.Module):
    def __init__(self, ksize=3):
        super().__init__()
        self.k = ksize

    def forward(self, x):
        w = torch.ones((x.size(1), 1, self.k, self.k), device=x.device) / (self.k ** 2)
        return F.conv2d(x, w, padding=self.k // 2, groups=x.size(1))