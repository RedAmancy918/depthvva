from typing import Dict, Tuple, Union, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class ImageStateCrossAttentionFusion(nn.Module):
    """
    Cross-Attention 融合模块：将 RGB+热力灰度图 concat 后的特征与 state 做 cross-attention

    
    Query: RGB+热力灰度图 concat 后的特征
    Key/Value: state 特征
    """
    
    def __init__(
        self,
        image_feat_dim: int,  # RGB+热力灰度图 concat 后的特征维度
        state_dim: int,       # state 特征维度
        fused_dim: int,      # 融合后的特征维度
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.image_feat_dim = image_feat_dim
        self.state_dim = state_dim
        self.fused_dim = fused_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        
        # Query 来自图像特征 (RGB+热力灰度图 concat)
        self.to_q = nn.Linear(image_feat_dim, inner_dim, bias=False)
        
        # Key/Value 来自 state
        self.to_k = nn.Linear(state_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(state_dim, inner_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,  # (B, image_feat_dim)
        state_features: torch.Tensor,  # (B, state_dim)
    ) -> torch.Tensor:
        """
        Args:
            image_features: RGB+热力灰度图 concat 后的特征 (B, image_feat_dim)
            state_features: state 特征 (B, state_dim)
        
        Returns:
            fused_features: 融合后的特征 (B, fused_dim)
        """
        B = image_features.shape[0]
        
        # Query 来自图像特征
        q = self.to_q(image_features)  # (B, inner_dim)
        q = rearrange(q, 'b (h d) -> b h 1 d', h=self.num_heads)  # (B, H, 1, D)
        
        # Key/Value 来自 state
        k = self.to_k(state_features)  # (B, inner_dim)
        v = self.to_v(state_features)  # (B, inner_dim)
        k = rearrange(k, 'b (h d) -> b h 1 d', h=self.num_heads)  # (B, H, 1, D)
        v = rearrange(v, 'b (h d) -> b h 1 d', h=self.num_heads)  # (B, H, 1, D)
        
        # 计算 attention scores
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (B, H, 1, 1)
        attn = F.softmax(attn, dim=-1)
        
        # 应用 attention 到 values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # (B, H, 1, D)
        out = rearrange(out, 'b h 1 d -> b (h d)')  # (B, inner_dim)
        
        # 输出投影
        out = self.to_out(out)  # (B, fused_dim)
        
        return out


class MultiImageObsEncoder(ModuleAttrMixin):
    """
    支持 Cross-Attention 融合的 MultiImageObsEncoder
    
    架构：
    1. RGB 和热力灰度图（head_camera_depth）分别通过 ResNet 编码
    2. 将编码后的特征 concat
    3. 提取 state 特征
    4. 使用 cross-attention 将 concat 后的图像特征与 state 融合
    5. 输出融合后的特征作为条件输入
    
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        # Cross-attention fusion parameters
        cross_attn_fused_dim: Optional[int] = None,
        cross_attn_num_heads: int = 8,
        cross_attn_head_dim: int = 64,
        cross_attn_dropout: float = 0.1,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        # ✅ 自动从 shape_meta 提取真实存在的 RGB keys
        self.rgb_keys = [k for k, v in self.key_shape_map.items() if v and len(v) == 3]
        
        # Cross-attention fusion parameters
        self.cross_attn_fusion = None
        self.cross_attn_fused_dim = cross_attn_fused_dim
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_head_dim = cross_attn_head_dim
        self.cross_attn_dropout = cross_attn_dropout

    def forward(self, obs_dict):
        batch_size = None
        image_features = list()  # RGB 和热力灰度图的特征
        state_features = list()   # state 特征
        
        # process rgb input (RGB 和热力灰度图)
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            image_features.append(feature)
        #else:
            # run each rgb obs to independent models
            # for key in self.rgb_keys:
            #     img = obs_dict[key]
            #     if batch_size is None:
            #         batch_size = img.shape[0]
            #     else:
            #         assert batch_size == img.shape[0]
            #     assert img.shape[1:] == self.key_shape_map[key]
            #     img = self.key_transform_map[key](img)
            #     feature = self.key_model_map[key](img)
            #     image_features.append(feature)
            # run each rgb obs to independent models (并行处理每个输入，使用不同模型)
        else:
        # run each rgb obs to independent models (并行处理每个输入，使用不同模型)
            assert len(self.rgb_keys) == len(self.key_model_map), \
                f"Expected same number of rgb_keys and key_model_map entries, got {len(self.rgb_keys)} vs {len(self.key_model_map)}"
            
            # Step 1: 预处理所有图像
            processed_imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]

                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                processed_imgs.append(img)

            # Step 2: 使用 CUDA Streams 并行处理
            streams = {key: torch.cuda.Stream() for key in self.rgb_keys}
            feature_dict = {}

            # 启动每个模型的 forward，使用不同的 stream
            for key in self.rgb_keys:
                with torch.cuda.stream(streams[key]):
                    feature_dict[key] = self.key_model_map[key](processed_imgs[self.rgb_keys.index(key)])

            # 等待所有 stream 完成
            torch.cuda.synchronize()

            # Step 3: 处理每个分支的输出（在所有模型都完成后）
            for key in self.rgb_keys:
                image_features.append(feature_dict[key])

        # process lowdim input (state)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            state_features.append(data)

        # 1. 将 RGB 和热力灰度图特征 concat
        image_feat_concat = torch.cat(image_features, dim=-1)  # (B, image_feat_dim)
        
        # 2. 将 state 特征 concat
        state_feat_concat = torch.cat(state_features, dim=-1) if state_features else None  # (B, state_dim)
        
        # 3. 动态创建 cross-attention 模块（如果还没有创建）
        if self.cross_attn_fusion is None:
            image_feat_dim = image_feat_concat.shape[-1]
            state_feat_dim = state_feat_concat.shape[-1] if state_feat_concat is not None else 0
            
            if state_feat_dim == 0:
                raise ValueError("No state features found for cross-attention fusion")
            
            # 如果未指定 fused_dim，使用 image_feat_dim
            fused_dim = self.cross_attn_fused_dim if self.cross_attn_fused_dim is not None else image_feat_dim
            
            self.cross_attn_fusion = ImageStateCrossAttentionFusion(
                image_feat_dim=image_feat_dim,
                state_dim=state_feat_dim,
                fused_dim=fused_dim,
                num_heads=self.cross_attn_num_heads,
                head_dim=self.cross_attn_head_dim,
                dropout=self.cross_attn_dropout,
            ).to(image_feat_concat.device)
            # 将 cross-attention 模块注册为子模块，以便参数能被正确管理
            self.add_module('cross_attn_fusion', self.cross_attn_fusion)
        
        # 4. Cross-attention 融合
        result = self.cross_attn_fusion(image_feat_concat, state_feat_concat)  # (B, fused_dim)
        
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((batch_size, ) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

