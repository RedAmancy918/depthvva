"""
Cross Attention模块，用于替代FiLM调制机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


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
            context: 条件特征
                - 格式1: (B, context_dim) - 单个全局条件
                - 格式2: (B, T, context_dim) - 时序条件（来自 Self-Attention 编码器）
        
        Returns:
            out: 注意力调制后的特征 (B, C, T)
        """
        B, C, T = x.shape
        
        # 转换为 (B, T, C) 以便做attention
        x = rearrange(x, 'b c t -> b t c')
        
        # 处理 context 格式
        if len(context.shape) == 2:
            # 格式1: (B, context_dim) -> (B, 1, context_dim)
            context = context.unsqueeze(1)
        elif len(context.shape) == 3:
            # 格式2: (B, T, context_dim) - 已经是正确格式
            # 如果 T 不匹配，需要处理（这里假设 T 匹配）
            pass
        else:
            raise ValueError(f"Unsupported context shape: {context.shape}")
        
        # 计算Q, K, V
        q = self.to_q(x)  # (B, T, inner_dim)
        k = self.to_k(context)  # (B, T_ctx, inner_dim) 或 (B, 1, inner_dim)
        v = self.to_v(context)  # (B, T_ctx, inner_dim) 或 (B, 1, inner_dim)
        
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
    
    支持三个模态：RGB、热力图、状态向量
    每个模态分别进行 Cross-Attention，然后融合
    
    例如：分别对RGB特征、热力图特征、关节状态做attention
    """
    
    def __init__(
        self,
        query_dim,
        context_dims,  # 列表：[rgb_dim, heatmap_dim, state_dim]
        num_heads=8,
        head_dim=64,
        dropout=0.0,
        use_feature_alignment=True,  # 是否使用特征对齐
        alignment_dim=None,  # 对齐后的维度，如果为 None 则使用 query_dim
        fusion_method='concat',  # 'concat' 或 'weighted_sum'
    ):
        super().__init__()
        
        self.num_modalities = len(context_dims)
        self.use_feature_alignment = use_feature_alignment
        self.fusion_method = fusion_method
        
        # 特征对齐维度
        if alignment_dim is None:
            alignment_dim = query_dim
        self.alignment_dim = alignment_dim
        
        # 为每个模态创建 Cross-Attention
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
        
        # 特征对齐投影层（如果需要）
        if use_feature_alignment:
            self.alignment_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(ctx_dim, alignment_dim),
                    nn.LayerNorm(alignment_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for ctx_dim in context_dims
            ])
        else:
            self.alignment_projs = None
        
        # 融合多个attention输出
        if fusion_method == 'concat':
            # Concat 方式：拼接所有 attention 输出
            self.fusion = nn.Sequential(
                nn.Linear(query_dim * len(context_dims), query_dim),
                nn.LayerNorm(query_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_method == 'weighted_sum':
            # 加权求和方式
            self.fusion_weights = nn.Parameter(torch.ones(len(context_dims)) / len(context_dims))
            self.fusion = nn.Sequential(
                nn.Linear(query_dim, query_dim),
                nn.LayerNorm(query_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        
    def forward(self, x, contexts):
        """
        Args:
            x: UNet特征 (B, C, T)
            contexts: 条件特征列表
                - 格式1: [(B, ctx_dim1), (B, ctx_dim2), ...] - 全局条件
                - 格式2: [(B, T, ctx_dim1), (B, T, ctx_dim2), ...] - 时序条件（来自 Self-Attention）
        
        Returns:
            out: 注意力调制后的特征 (B, C, T)
        """
        if len(contexts) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} contexts, got {len(contexts)}"
            )
        
        # 处理每个模态的条件特征
        processed_contexts = []
        for i, ctx in enumerate(contexts):
            if ctx is None:
                # 如果某个模态不存在，跳过
                processed_contexts.append(None)
                continue
            
            # 特征对齐（如果需要）
            if self.use_feature_alignment and self.alignment_projs is not None:
                # 处理不同格式的 context
                if len(ctx.shape) == 2:
                    # (B, ctx_dim) -> (B, alignment_dim)
                    ctx_aligned = self.alignment_projs[i](ctx)
                elif len(ctx.shape) == 3:
                    # (B, T, ctx_dim) -> (B, T, alignment_dim)
                    B, T, D = ctx.shape
                    ctx_flat = ctx.reshape(B * T, D)
                    ctx_aligned_flat = self.alignment_projs[i](ctx_flat)
                    ctx_aligned = ctx_aligned_flat.reshape(B, T, -1)
                else:
                    raise ValueError(f"Unsupported context shape: {ctx.shape}")
                processed_contexts.append(ctx_aligned)
            else:
                processed_contexts.append(ctx)
        
        # 对每个模态分别进行 Cross-Attention
        attn_outs = []
        for i, (attn, ctx) in enumerate(zip(self.cross_attentions, processed_contexts)):
            if ctx is None:
                # 如果某个模态不存在，使用零特征
                B, C, T = x.shape
                attn_out = torch.zeros_like(x)
            else:
                attn_out = attn(x, ctx)
            attn_outs.append(attn_out)
        
        # 融合多个 attention 输出
        if self.fusion_method == 'concat':
            # Concat + 融合
            attn_concat = torch.cat(attn_outs, dim=1)  # (B, C*N, T)
            attn_concat = rearrange(attn_concat, 'b c t -> b t c')
            out = self.fusion(attn_concat)  # (B, T, query_dim)
            out = rearrange(out, 'b t c -> b c t')  # (B, query_dim, T)
        elif self.fusion_method == 'weighted_sum':
            # 加权求和
            # 归一化权重
            weights = F.softmax(self.fusion_weights, dim=0)
            
            # 加权求和
            out = sum(w * attn_out for w, attn_out in zip(weights, attn_outs))
            
            # 融合层
            out = rearrange(out, 'b c t -> b t c')
            out = self.fusion(out)
            out = rearrange(out, 'b t c -> b c t')
        
        return out


# 为了方便导入
__all__ = [
    'CrossAttention1D',
    'ConditionalResidualBlock1DWithCrossAttention',
    'MultiHeadCrossAttention1D',
]

