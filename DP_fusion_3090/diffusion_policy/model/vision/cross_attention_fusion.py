"""
Cross-Attention 融合模块
用于融合多个模态的特征（RGB、热力图、状态向量）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class MultiModalCrossAttentionFusion(nn.Module):
    """
    多模态 Cross-Attention 融合模块
    
    用于融合三个模态的特征（RGB、热力图、状态向量）
    使用 RGB 作为 Query，热力图和状态向量作为 Key/Value
    
    Args:
        rgb_dim: RGB 特征维度
        heatmap_dim: 热力图特征维度
        state_dim: 状态向量维度
        fused_dim: 融合后的特征维度
        num_heads: 注意力头数
        head_dim: 每个头的维度
        dropout: Dropout 比率
        use_feature_alignment: 是否使用特征对齐
        alignment_dim: 对齐后的维度，如果为 None 则使用 fused_dim
    """
    
    def __init__(
        self,
        rgb_dim: int,
        heatmap_dim: int,
        state_dim: int,
        fused_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        use_feature_alignment: bool = True,
        alignment_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.rgb_dim = rgb_dim
        self.heatmap_dim = heatmap_dim
        self.state_dim = state_dim
        self.fused_dim = fused_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_feature_alignment = use_feature_alignment
        
        # 特征对齐维度
        if alignment_dim is None:
            alignment_dim = fused_dim
        self.alignment_dim = alignment_dim
        
        # 特征对齐投影层（如果需要）
        if use_feature_alignment:
            self.rgb_alignment = nn.Sequential(
                nn.Linear(rgb_dim, alignment_dim),
                nn.LayerNorm(alignment_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # 对齐后的维度
            aligned_rgb_dim = alignment_dim
            
            # 热力图对齐（如果存在）
            if heatmap_dim > 0:
                self.heatmap_alignment = nn.Sequential(
                    nn.Linear(heatmap_dim, alignment_dim),
                    nn.LayerNorm(alignment_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                aligned_heatmap_dim = alignment_dim
            else:
                self.heatmap_alignment = None
                aligned_heatmap_dim = 0
            
            # 状态向量对齐（如果存在）
            if state_dim > 0:
                self.state_alignment = nn.Sequential(
                    nn.Linear(state_dim, alignment_dim),
                    nn.LayerNorm(alignment_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                aligned_state_dim = alignment_dim
            else:
                self.state_alignment = None
                aligned_state_dim = 0
        else:
            self.rgb_alignment = None
            self.heatmap_alignment = None
            self.state_alignment = None
            aligned_rgb_dim = rgb_dim
            aligned_heatmap_dim = heatmap_dim
            aligned_state_dim = state_dim
        
        # 融合 Query（使用 RGB 特征）
        inner_dim = head_dim * num_heads
        self.to_q = nn.Linear(aligned_rgb_dim, inner_dim, bias=False)
        
        # 融合 Key/Value（使用热力图和状态向量）
        # 热力图 Key/Value（如果存在）
        if aligned_heatmap_dim > 0:
            self.to_k_heatmap = nn.Linear(aligned_heatmap_dim, inner_dim, bias=False)
            self.to_v_heatmap = nn.Linear(aligned_heatmap_dim, inner_dim, bias=False)
        else:
            self.to_k_heatmap = None
            self.to_v_heatmap = None
        
        # 状态向量 Key/Value（如果存在）
        if aligned_state_dim > 0:
            self.to_k_state = nn.Linear(aligned_state_dim, inner_dim, bias=False)
            self.to_v_state = nn.Linear(aligned_state_dim, inner_dim, bias=False)
        else:
            self.to_k_state = None
            self.to_v_state = None
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.scale = head_dim ** -0.5
    
    def forward(
        self,
        rgb_features: torch.Tensor,  # (B, T, rgb_dim)
        heatmap_features: Optional[torch.Tensor] = None,  # (B, T, heatmap_dim)
        state_features: Optional[torch.Tensor] = None,  # (B, T, state_dim)
    ) -> torch.Tensor:
        """
        Args:
            rgb_features: RGB 特征 (B, T, rgb_dim)
            heatmap_features: 热力图特征 (B, T, heatmap_dim) 或 None
            state_features: 状态向量特征 (B, T, state_dim) 或 None
        
        Returns:
            fused_features: 融合后的特征 (B, T, fused_dim)
        """
        B, T, _ = rgb_features.shape
        
        # 特征对齐
        if self.use_feature_alignment:
            rgb_aligned = self.rgb_alignment(rgb_features)  # (B, T, alignment_dim)
            heatmap_aligned = self.heatmap_alignment(heatmap_features) if heatmap_features is not None else None
            state_aligned = self.state_alignment(state_features) if state_features is not None else None
        else:
            rgb_aligned = rgb_features
            heatmap_aligned = heatmap_features
            state_aligned = state_features
        
        # Query 来自 RGB
        q = self.to_q(rgb_aligned)  # (B, T, inner_dim)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)  # (B, H, T, D)
        
        # 收集所有 Key/Value
        keys = []
        values = []
        
        # 热力图 Key/Value
        if heatmap_aligned is not None and self.to_k_heatmap is not None:
            k_heatmap = self.to_k_heatmap(heatmap_aligned)  # (B, T, inner_dim)
            v_heatmap = self.to_v_heatmap(heatmap_aligned)  # (B, T, inner_dim)
            k_heatmap = rearrange(k_heatmap, 'b t (h d) -> b h t d', h=self.num_heads)
            v_heatmap = rearrange(v_heatmap, 'b t (h d) -> b h t d', h=self.num_heads)
            keys.append(k_heatmap)
            values.append(v_heatmap)
        
        # 状态向量 Key/Value
        if state_aligned is not None and self.to_k_state is not None:
            k_state = self.to_k_state(state_aligned)  # (B, T, inner_dim)
            v_state = self.to_v_state(state_aligned)  # (B, T, inner_dim)
            k_state = rearrange(k_state, 'b t (h d) -> b h t d', h=self.num_heads)
            v_state = rearrange(v_state, 'b t (h d) -> b h t d', h=self.num_heads)
            keys.append(k_state)
            values.append(v_state)
        
        # 如果没有其他模态，只使用 RGB（自注意力）
        if len(keys) == 0:
            # 使用 RGB 自己作为 Key/Value
            if self.to_k_heatmap is not None:
                k_rgb = self.to_k_heatmap(rgb_aligned)
                v_rgb = self.to_v_heatmap(rgb_aligned)
            elif self.to_k_state is not None:
                k_rgb = self.to_k_state(rgb_aligned)
                v_rgb = self.to_v_state(rgb_aligned)
            else:
                # 如果没有其他模态的投影层，创建一个临时的
                k_rgb = self.to_q(rgb_aligned)  # 使用 Query 投影
                v_rgb = self.to_q(rgb_aligned)
            
            k_rgb = rearrange(k_rgb, 'b t (h d) -> b h t d', h=self.num_heads)
            v_rgb = rearrange(v_rgb, 'b t (h d) -> b h t d', h=self.num_heads)
            keys.append(k_rgb)
            values.append(v_rgb)
        
        # 拼接所有 Key/Value
        k = torch.cat(keys, dim=2)  # (B, H, T_total, D) 其中 T_total = T * num_modalities
        v = torch.cat(values, dim=2)  # (B, H, T_total, D)
        
        # 计算 Attention
        attn = torch.einsum('b h t d, b h n d -> b h t n', q, k) * self.scale  # (B, H, T, T_total)
        attn = F.softmax(attn, dim=-1)
        
        # 应用 Attention 到 Values
        out = torch.einsum('b h t n, b h n d -> b h t d', attn, v)  # (B, H, T, D)
        out = rearrange(out, 'b h t d -> b t (h d)')  # (B, T, inner_dim)
        
        # 输出投影
        out = self.to_out(out)  # (B, T, fused_dim)
        
        return out


# 为了方便导入
__all__ = [
    'MultiModalCrossAttentionFusion',
]

