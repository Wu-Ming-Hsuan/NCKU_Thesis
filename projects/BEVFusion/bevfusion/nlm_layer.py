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

    def __init__(self, in_channels=None, embed=True, softmax=True, zero_init=True):
        super(NonLocalDenoising, self).__init__()
        self.embed = embed
        self.softmax = softmax
        self.zero_init = zero_init
        
        # If in_channels is None, we'll determine it dynamically during forward
        # This makes it more flexible for different feature layers
        self.in_channels = in_channels
        
        # Initialize learnable parameters for the embedding functions
        if self.embed:
            self.theta_conv = None
            self.phi_conv = None
        
        # Final projection layer
        self.projection = None
        
    def _init_layers(self, n_in, device):
        """Initialize layers with the correct dimensions."""
        self.in_channels = n_in
        
        if self.embed:
            embed_dim = max(n_in // 2, 1)
            self.theta_conv = nn.Conv2d(n_in, embed_dim, 1, 1, 0, bias=False).to(device)
            self.phi_conv = nn.Conv2d(n_in, embed_dim, 1, 1, 0, bias=False).to(device)
            nn.init.normal_(self.theta_conv.weight, std=0.01)
            nn.init.normal_(self.phi_conv.weight, std=0.01)
        
        # Final projection layer to match input channels
        self.projection = nn.Conv2d(n_in, n_in, 1, 1, 0, bias=False).to(device)
        
        # Initialize gamma to 0 if using zero_init
        if self.zero_init:
            nn.init.zeros_(self.projection.weight)
        else:
            nn.init.normal_(self.projection.weight, std=0.01)
    
    def non_local_op(self, x):
        """
        Apply the non-local operation.
        
        Args:
            x (torch.Tensor): Input feature maps [N, C, H, W].
        
        Returns:
            torch.Tensor: Denoised feature maps with same shape as input.
        """
        n, c, h, w = x.shape
        
        if self.embed:
            # Compute embedded version of input
            theta = self.theta_conv(x)  # [N, C/2, H, W]
            phi = self.phi_conv(x)      # [N, C/2, H, W]
            g = x                      # [N, C, H, W]
        else:
            theta, phi, g = x, x, x
        
        # Determine whether to use efficient computation based on feature dimensions
        if c > h * w or self.softmax:
            # Standard non-local block computation
            # Reshape for matrix multiplication
            theta = theta.view(n, theta.shape[1], -1)  # [N, C', H*W]
            phi = phi.view(n, phi.shape[1], -1)        # [N, C', H*W]
            g = g.view(n, g.shape[1], -1)              # [N, C, H*W]
            
            # Transpose for matrix multiplication
            theta = theta.permute(0, 2, 1)  # [N, H*W, C']
            
            # Matrix multiplication and softmax normalization
            f = torch.matmul(theta, phi)  # [N, H*W, H*W]
            
            if self.softmax:
                # Apply scaling factor
                f = f / torch.sqrt(torch.tensor(theta.shape[-1], dtype=theta.dtype, device=theta.device))
                f = F.softmax(f, dim=-1)  # [N, H*W, H*W]
            
            # Matrix multiplication with g
            y = torch.matmul(f, g.permute(0, 2, 1))  # [N, H*W, C]
            y = y.permute(0, 2, 1).contiguous()  # [N, C, H*W]
            y = y.view(n, c, h, w)  # [N, C, H, W]
        else:
            # More memory efficient computation for large feature maps
            phi = phi.view(n, phi.shape[1], -1)  # [N, C, H*W]
            g = g.view(n, g.shape[1], -1)        # [N, C, H*W]
            
            # First matrix multiplication
            f = torch.matmul(phi, g.transpose(-1, -2))  # [N, C, C]
            
            # Second matrix multiplication
            theta = theta.view(n, theta.shape[1], -1)  # [N, C, H*W]
            y = torch.matmul(f, theta)  # [N, C, H*W]
            
            # Reshape back to original dimensions
            y = y.view(n, c, h, w)  # [N, C, H, W]
            
            # Normalize by number of positions
            y = y / (h * w)
        
        return y
    
    def forward(self, x):
        """
        Forward function.
        
        Args:
            x (torch.Tensor): Input feature maps [N, C, H, W].
        
        Returns:
            torch.Tensor: Denoised feature maps with same shape as input.
        """
        # Initialize layers if not done yet or if channels changed
        if self.in_channels != x.shape[1] or self.projection is None:
            self._init_layers(x.shape[1], x.device)
        
        # Apply non-local operation
        residual = self.non_local_op(x)
        
        # Apply final projection 
        residual = self.projection(residual)
        
        # Add residual connection (identity mapping)
        output = x + residual
        
        return output 