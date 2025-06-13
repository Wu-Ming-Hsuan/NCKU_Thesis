import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, embed_dim=64):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, C, x_size[0], x_size[1])


class FractalBlockWithWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=2., norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        assert N == H * W, f"Expected N=H×W but got N={N}, H={H}, W={W}"

        shortcut = x
        x = self.norm1(x).transpose(1, 2).view(B, C, H, W)

        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        Hp, Wp = x.shape[-2:]

        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

        x, _ = self.attn(x, x, x)

        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
        x = x[:, :, :H, :W].flatten(2).transpose(1, 2)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register_module()
class FractalEnhancer(nn.Module):
    def __init__(
        self,
        in_channels=256,
        img_size=(180, 180),
        patch_size=4,
        embed_dim=256,
        depths=(2, 2),
        num_heads=(4, 4),
        window_size=15,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patches_resolution = self.patch_embed.patches_resolution
        self.layers = nn.ModuleList([
            FractalBlockWithWindowAttention(
                dim=embed_dim,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=2.0,
                norm_layer=norm_layer
            )
            for i in range(len(depths))
            for _ in range(depths[i])
        ])
        self.norm = norm_layer(embed_dim)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_size = self.patches_resolution

        x_embed = self.patch_embed(x)  # [B, N, C]
        for layer in self.layers:
            x_embed = layer(x_embed, x_size[0], x_size[1])

        x_embed = self.norm(x_embed)
        x_out = self.patch_unembed(x_embed, x_size)

        x_out = F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)
        return x_out
