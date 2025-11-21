"""
带Cross Attention的Conditional UNet1D
"""
from typing import Union
import logging
import torch
import torch.nn as nn
import einops

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d,
    Upsample1d,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.cross_attention_components import (
    ConditionalResidualBlock1DWithCrossAttention
)

logger = logging.getLogger(__name__)


class ConditionalUnet1DWithCrossAttention(nn.Module):
    """
    使用Cross Attention替代FiLM的Conditional UNet1D
    """
    
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        use_cross_attention=True,
        num_attention_heads=8,
        attention_head_dim=64,
    ):
        super().__init__()
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Diffusion step编码器
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # 条件维度（用于Cross Attention的context）
        # 如果使用Cross Attention，条件不再与timestep concat
        if use_cross_attention:
            cond_dim = global_cond_dim if global_cond_dim is not None else 0
            # timestep embedding单独处理
            self.time_cond_dim = dsed
        else:
            # FiLM模式：timestep和条件concat
            cond_dim = dsed
            if global_cond_dim is not None:
                cond_dim += global_cond_dim
            self.time_cond_dim = cond_dim
        
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        # Local条件编码器（如果有）
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_in, dim_out,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_in, dim_out,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
            ])
        
        # 中间层
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1DWithCrossAttention(
                mid_dim, mid_dim,
                cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                use_cross_attention=use_cross_attention,
            ),
            ConditionalResidualBlock1DWithCrossAttention(
                mid_dim, mid_dim,
                cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                use_cross_attention=use_cross_attention,
            ),
        ])
        
        # Downsampling层
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_in, dim_out,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_out, dim_out,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
        
        # Upsampling层
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_out * 2, dim_in,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
                ConditionalResidualBlock1DWithCrossAttention(
                    dim_in, dim_in,
                    cond_dim=cond_dim if use_cross_attention else self.time_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    use_cross_attention=use_cross_attention,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))
        
        from diffusion_policy.model.diffusion.conv1d_components import Conv1dBlock
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        
        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.use_cross_attention = use_cross_attention
        
        logger.info(
            f"ConditionalUnet1DWithCrossAttention: use_cross_attention={use_cross_attention}, "
            f"parameters: {sum(p.numel() for p in self.parameters()):,}"
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        Args:
            sample: (B, T, input_dim)
            timestep: (B,) or int
            local_cond: (B, T, local_cond_dim) [可选]
            global_cond: (B, global_cond_dim) [可选]
        
        Returns:
            output: (B, T, input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')
        
        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        
        time_emb = self.diffusion_step_encoder(timesteps)
        
        # 2. 准备条件
        if self.use_cross_attention:
            # Cross Attention模式：global_cond作为context
            context = global_cond
        else:
            # FiLM模式：concat timestep和global_cond
            if global_cond is not None:
                context = torch.cat([time_emb, global_cond], dim=-1)
            else:
                context = time_emb
        
        # 3. 编码local条件（如果有）
        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, context)
            h_local.append(x)
            x = resnet2(local_cond, context)
            h_local.append(x)
        
        # 4. Downsampling
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, context)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, context)
            h.append(x)
            x = downsample(x)
        
        # 5. Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, context)
        
        # 6. Upsampling
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, context)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, context)
            x = upsample(x)
        
        # 7. Final
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        
        return x

