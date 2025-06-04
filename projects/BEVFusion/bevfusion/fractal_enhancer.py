import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

# --- 1. 金字塔patch embed/unembed ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, C
        x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x

# --- 2. Fractal Attention Block（可做多層） ---
class FractalBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=2., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
    def forward(self, x):
        # x: [B, N, C]
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# --- 3. FractalDRCTv4 Feature Pyramid (極簡單金字塔多層堆疊) ---
@MODELS.register_module()
class FractalEnhancer(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size=(200, 200),
        patch_size=4,
        embed_dim=64,
        depths=(2, 2),
        num_heads=(4, 4),
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.layers = nn.ModuleList([
            FractalBlock(embed_dim, num_heads[i], window_size=7, mlp_ratio=2., norm_layer=norm_layer)
            for i in range(len(depths)) for _ in range(depths[i])
        ])
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim, norm_layer=norm_layer)
        self.norm = norm_layer(embed_dim)
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, C, H, W)
        x_size = self.patches_resolution
        x = self.patch_embed(x)         # B, N, C
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x
