"""
Cross Attention模块，用于替代FiLM调制机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention1D(nn.Module):
    """
    1D Cross Attention模块
    
    Query: 来自UNet的特征 (B, C, T)
    Key/Value: 来自观测的条件 (B, cond_dim)
    """
    
    def __init__(
        self,
        query_dim,
        context_dim,
        num_heads=8,
        head_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Query从UNet特征投影
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Key和Value从条件投影
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context):
        """
        Args:
            x: UNet特征 (B, C, T)
            context: 条件特征 (B, context_dim)
        
        Returns:
            out: 注意力调制后的特征 (B, C, T)
        """
        B, C, T = x.shape
        
        # 转换为 (B, T, C) 以便做attention
        x = rearrange(x, 'b c t -> b t c')
        
        # 扩展context维度以匹配时间步
        # context: (B, context_dim) -> (B, 1, context_dim)
        context = context.unsqueeze(1)
        
        # 计算Q, K, V
        q = self.to_q(x)  # (B, T, inner_dim)
        k = self.to_k(context)  # (B, 1, inner_dim)
        v = self.to_v(context)  # (B, 1, inner_dim)
        
        # 重塑为多头
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # 计算attention scores
        attn = torch.einsum('b h t d, b h n d -> b h t n', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用attention到values
        out = torch.einsum('b h t n, b h n d -> b h t d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        # 输出投影
        out = self.to_out(out)
        
        # 转回 (B, C, T)
        out = rearrange(out, 'b t c -> b c t')
        
        return out


class ConditionalResidualBlock1DWithCrossAttention(nn.Module):
    """
    带Cross Attention的1D条件残差块
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        num_attention_heads=8,
        attention_head_dim=64,
        use_cross_attention=True,
    ):
        super().__init__()
        
        from diffusion_policy.model.diffusion.conv1d_components import Conv1dBlock
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # Cross Attention模块
            self.cross_attn = CrossAttention1D(
                query_dim=out_channels,
                context_dim=cond_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
            )
            
            # Layer Norm
            self.norm = nn.GroupNorm(n_groups, out_channels)
        else:
            # 降级到FiLM
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        
        # 残差连接
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
        
    def forward(self, x, cond):
        """
        Args:
            x: (B, C, T)
            cond: (B, cond_dim)
        
        Returns:
            out: (B, C, T)
        """
        out = self.blocks[0](x)
        
        if self.use_cross_attention:
            # Cross Attention分支
            # 1. Normalize
            out_norm = self.norm(out)
            
            # 2. Cross Attention
            attn_out = self.cross_attn(out_norm, cond)
            
            # 3. 残差连接
            out = out + attn_out
        else:
            # FiLM分支（向后兼容）
            embed = self.cond_encoder(cond)
            out = out + embed
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        
        return out


class MultiHeadCrossAttention1D(nn.Module):
    """
    多头Cross Attention，支持多个条件源
    
    例如：分别对RGB特征、热力图特征、关节状态做attention
    """
    
    def __init__(
        self,
        query_dim,
        context_dims,  # 列表：[rgb_dim, heatmap_dim, agent_pos_dim]
        num_heads=8,
        head_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        
        self.cross_attentions = nn.ModuleList([
            CrossAttention1D(
                query_dim=query_dim,
                context_dim=ctx_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
            for ctx_dim in context_dims
        ])
        
        # 融合多个attention输出
        self.fusion = nn.Linear(query_dim * len(context_dims), query_dim)
        
    def forward(self, x, contexts):
        """
        Args:
            x: (B, C, T)
            contexts: 列表 [(B, ctx_dim1), (B, ctx_dim2), ...]
        
        Returns:
            out: (B, C, T)
        """
        attn_outs = []
        for attn, ctx in zip(self.cross_attentions, contexts):
            attn_out = attn(x, ctx)
            attn_outs.append(attn_out)
        
        # Concat + 融合
        attn_concat = torch.cat(attn_outs, dim=1)  # (B, C*N, T)
        attn_concat = rearrange(attn_concat, 'b c t -> b t c')
        out = self.fusion(attn_concat)
        out = rearrange(out, 'b t c -> b c t')
        
        return out


# 为了方便导入
__all__ = [
    'CrossAttention1D',
    'ConditionalResidualBlock1DWithCrossAttention',
    'MultiHeadCrossAttention1D',
]

