import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class NonLocalDenoising(nn.Module):
    """
    Implementation of Non-Local Mean denoising layer based on Feature Denoising for adversarial robustness.
    
    Args:
        in_channels (int): Input channels.
        embed (bool): Whether to use embedding on theta & phi. Default: True.
        softmax (bool): Whether to use gaussian (softmax) version or the dot-product version. Default: True.
        zero_init (bool): Whether to initialize gamma to zero for residual learning. Default: True.
    """

    def __init__(self,
                 in_channels: int = None,
                 embed: bool = True,
                 softmax: bool = True,
                 zero_init: bool = True):
        super().__init__()
        # configuration
        self.embed = embed
        self.softmax = softmax
        self.zero_init = zero_init
        self.in_channels = in_channels

        # 先放 placeholder，真正的層等 _build_layers 建好後替換
        self.theta_conv = nn.Identity()
        self.phi_conv   = nn.Identity()
        self.projection = nn.Identity()

        # 若已知 in_channels，建層並初始化；否則延後到第一次 forward
        if in_channels is not None:
            self._build_layers(in_channels)

    def _build_layers(self, n_in: int):
        """在 Module 下建立/替換真正的 conv 層並初始化權重。"""
        assert n_in > 0, 'in_channels 應為正整數'
        self.in_channels = n_in

        # θ, φ 嵌入 (C → C')，C' = C // 2，最少 1
        if self.embed:
            c_embed = max(n_in // 2, 1)
            self.theta_conv = nn.Conv2d(n_in, c_embed, kernel_size=1, bias=False)
            self.phi_conv   = nn.Conv2d(n_in, c_embed, kernel_size=1, bias=False)
            nn.init.normal_(self.theta_conv.weight, std=0.01)
            nn.init.normal_(self.phi_conv.weight,   std=0.01)
        else:
            self.theta_conv = self.phi_conv = nn.Identity()

        # γ 投影 (C → C)
        self.projection = nn.Conv2d(n_in, n_in, kernel_size=1, bias=False)
        if self.zero_init:
            nn.init.zeros_(self.projection.weight)
        else:
            nn.init.normal_(self.projection.weight, std=0.01)

        # 將新建層註冊到 Module（覆蓋舊的 Identity）
        self.add_module('theta_conv', self.theta_conv)
        self.add_module('phi_conv',   self.phi_conv)
        self.add_module('projection', self.projection)

        # 快取 softmax scale
        self.register_buffer('_scale', torch.tensor(c_embed if self.embed else n_in,
                                                    dtype=torch.float32))
        
    def init_weights(self):
        pass
    
    def _non_local_op(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, H, W] -> y: [N, C, H, W]"""
        n, c, h, w = x.shape

        # 取 θ, φ, g
        theta = self.theta_conv(x)                       # [N, Cθ, H, W]
        phi   = self.phi_conv(x)                         # [N, Cθ, H, W]
        g     = x                                        # [N, C,  H, W]

        # Flatten spatial (H*W)
        theta = theta.reshape(n, theta.shape[1], -1).transpose(1, 2).contiguous()  # [N, HW, Cθ]
        phi   =   phi.reshape(n,   phi.shape[1], -1)                               # [N, Cθ, HW]
        g     =     g.reshape(n,     g.shape[1], -1).transpose(1, 2).contiguous()   # [N, HW, C]

        # Affinity matrix f
        f = theta @ phi                      # [N, HW, HW]
        if self.softmax:
            f = f / torch.sqrt(self._scale)  # scale
            f = F.softmax(f, dim=-1)

        # Aggregate
        y = f @ g                            # [N, HW, C]
        y = y.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 若第一次進來或 channel 改變 → 建層
        if self.in_channels != x.shape[1] or isinstance(self.projection, nn.Identity):
            self._build_layers(x.shape[1])

        # 非在-place，避免覆蓋原張量
        residual = self._non_local_op(x)
        residual = self.projection(residual)
        return x + residual