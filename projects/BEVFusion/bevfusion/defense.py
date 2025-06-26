import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

####################################################################################################
# W NLM
####################################################################################################
@MODELS.register_module()
class WNLM(nn.Module):
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


####################################################################################################
# SAMS NLM
####################################################################################################


#########################
# Helper functions
#########################

def pad_to_multiple(x, multiple):
    """Pad H and W so they are divisible by ``multiple``."""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w

def unpad(x, pad_h, pad_w):
    """Remove padding added by pad_to_multiple."""
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :-pad_w]
    return x

############################
# Patch Embed / UnEmbed
############################

class PatchEmbed(nn.Module):
    """Flatten non‑overlapping patches to token sequence."""

    def __init__(self, img_size=224, patch_size=1, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.h_res = img_size[0] // patch_size[0]
        self.w_res = img_size[1] // patch_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

class PatchUnEmbed(nn.Module):
    """Restore (B,N,D) tokens to feature map."""

    def __init__(self, img_size=224, patch_size=1, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.h_res = img_size[0] // patch_size[0]
        self.w_res = img_size[1] // patch_size[1]
        self.embed_dim = embed_dim

    def forward(self, x):
        B, N, C = x.shape
        return x.transpose(1, 2).view(B, C, self.h_res, self.w_res)

############################
# Fractal‑NLM core module
############################

class SAMSNLMFilter(nn.Module):
    """Multi‑Scale + Symmetry Non‑Local module."""

    def __init__(self, dim=64, reduction=18, dilations=(1, 2, 3), token_batch=32, residual_scale=0.2):
        super().__init__()
        self.dim = dim
        self.residual_scale = residual_scale
        inner_dim = dim // reduction
        self.reduce = nn.Conv2d(dim, inner_dim, 1)
        self.dil_convs = nn.ModuleList(
            [nn.Conv2d(inner_dim, inner_dim, 3, padding=d, dilation=d) for d in dilations]
        )
        self.fuse = nn.Conv2d(inner_dim * len(dilations) * 6, dim, 1)
        self.token_batch = token_batch

    # six C4 symmetry transforms (identity, LR flip, UD flip, 90/180/270 rot)
    @staticmethod
    def _c4_transforms(x):
        return [
            x,
            torch.flip(x, [-1]),
            torch.flip(x, [-2]),
            torch.rot90(x, 1, (-2, -1)),
            torch.rot90(x, 2, (-2, -1)),
            torch.rot90(x, 3, (-2, -1)),
        ]

    def forward(self, x):
        B, C, H, W = x.shape
        x_red = self.reduce(x)  # (B, C//r, H, W)
        feats = []
        for conv in self.dil_convs:
            conv_out = conv(x_red)
            for tf in self._c4_transforms(x_red):
                feats.append(conv_out * tf)
        cat = torch.cat(feats, dim=1)
        fused = self.fuse(cat)
        tokens = fused.flatten(2).transpose(1, 2)
        if self.token_batch is None or tokens.shape[1] <= self.token_batch:
            attn = torch.softmax(tokens @ tokens.transpose(1, 2) / math.sqrt(C), dim=-1)
            out = attn @ tokens
        else:
            outs = []
            for chunk in tokens.split(self.token_batch, dim=1):
                attn_chunk = torch.softmax(chunk @ tokens.transpose(1, 2) / math.sqrt(C), dim=-1)
                outs.append(attn_chunk @ tokens)
            out = torch.cat(outs, dim=1)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + self.residual_scale * out

#########################################
# Full purifier (patch‑token‑patch flow)
#########################################

@MODELS.register_module()
class SAMSNLM(nn.Module):
    """End‑to‑end pre‑filter that wraps Patch⇄Token⇄Patch and Fractal‑NLM."""

    def __init__(self, in_chans=256, embed_dim=256, img_size=180, patch_size=1, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.nlm = SAMSNLMFilter(dim=embed_dim, **kwargs)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim)

    def forward(self, x):
        ori_size = x.shape[-2:]
        tokens = self.patch_embed(x)
        fmap = self.patch_unembed(tokens)
        cleaned = self.nlm(fmap)
        if cleaned.shape[-2:] != ori_size:
            cleaned = F.interpolate(cleaned, size=ori_size, mode="bilinear", align_corners=False)
        return cleaned