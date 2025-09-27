# depthvva
## 2025.9.14
- **把臂的端口名字固定为了 blue_follower、blue_leader**
- **之前连接深度相机的线竟然是usb2.1，换了一根usb3**
- **lerobot可以读取深度信息但是没有办法保存，目前考虑把realsense官方读取保存的代码嵌进lerobot的**
- **之前设置的realsense的帧率是15，所以最后的视频特别快，我就说为什么我上次采的深度数据是动作的一半。现在改回30了。**

数据采集脚本
```bash
python -m lerobot.record   --robot.disable_torque_on_disconnect=true         --robot.type=so101_follower     --robot.port=/dev/blue_follower         --robot.id=R12253310      --robot.cameras="{'handeye':{'type':'opencv','index_or_path':10,'width':640,'height':480,'fps':30},'fixed':{'type':'intelrealsense','serial_number_or_name':'250122073394','width':640,'height':480,'fps':30,'use_depth':true}}"    --teleop.type=so101_leader        --teleop.port=/dev/blue_leader  --teleop.id=R07253310   --display_data=true       --dataset.root="$HOME/datasets/lerobot/so101"   --dataset.push_to_hub=false       --dataset.repo_id="local/so101"         --dataset.num_episodes=10       --dataset.episode_time_s=20       --dataset.single_task='Place the tape'  --dataset.fps=30
```
## 2025.9.17
相机标定：

```py
 python calculate_ax.py \
  --images_glob "out_rs/images/*.png" \
  --calib out_rs/camera_calib.json \
  --board board.yaml \
  --out out_rs/cam_world_extrinsics.json
```
推理：
```bash
python -m lerobot.record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12253310 \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=R07253310 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras="{'handeye':{'type':'opencv','index_or_path':2,'width':640,'height':480,'fps':30},'fixed':{'type':'intelrealsense','serial_number_or_name':'250122073394','width':640,'height':480,'fps':30}}"  \
  --display_data=true \
  --dataset.single_task='Place the cube' \
  --policy.type=diffusion \
  --policy.path=/home/ff/ff00/depthvva/thirdparty/lerobot/outputs/train/diffusion_so101_exp/checkpoints/lastpretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=${HF_USER}/eval_so101 \
  --dataset.push_to_hub=false
  ```
- **设计了将方块夹如透明胶圈内的task，第一次采集了30组数据，在本地step=4000的情况下train了模型，并部署的so100中，轨迹大致合理**
- **数据量不够，再次采集了30组，但这次报错图像与动作帧数不匹配。所有图像的帧数都只有100帧**

##2025.9.18
- **修改/home/ff/ff00/depthvva/thirdparty/lerobot/src/lerobot/datasets/compute_stats.py 
如以下：解决了episodes_stats.jsonl中handeye和fixed中只保存100帧的问题**

```py
def sample_images(image_paths: list[str]) -> np.ndarray:
    images = []
    for path in image_paths:
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)
        images.append(img)
    return np.stack(images, axis=0)
```

-**电机坏了**


## 测试
**dp+depthany**
1. 主摄 RGB+深度热力 腕部 RGB+深度热力 path:
   图像拼接方式问题，裁减问题，推理输入源问题
2. 主摄 深度热力     腕部 深度热力 path:
   效果不错，但是很难看到
3. 主摄 RGB         腕部 深度热力 path:

我理解你的问题了！你想要在训练时增加一个摄像头的通道维度，而不是简单地将两个画面拼接在一起。这涉及到模型架构的修改，让我帮你找到需要修改的具体位置。


[3 tools called]

现在让我查看具体的RGB编码器实现：


[2 tools called]

现在让我查看具体的配置参数：


[2 tools called]

现在我明白了！让我为你详细解释如何修改代码来增加摄像头的通道维度，而不是简单拼接两个画面。

## 关键发现

通过分析代码，我发现LeRobot中的多摄像头处理有两种模式：

1. **共享编码器模式** (`use_separate_rgb_encoder_per_camera: false`) - 这是默认模式
2. **分离编码器模式** (`use_separate_rgb_encoder_per_camera: true`) - 每个摄像头使用独立的编码器

## 需要修改的关键位置


[1 tool called]

### 1. **修改DiffusionRgbEncoder以支持多通道输入**

在 `/home/rb01/Geo/depthvva/thirdparty/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py` 中：

```python
class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # ... 现有代码 ...
        
        # 修改这里：支持多通道输入
        images_shape = next(iter(config.image_features.values())).shape
        # 计算总通道数：每个摄像头的通道数 × 摄像头数量
        total_channels = images_shape[0] * len(config.image_features)
        
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, total_channels, *dummy_shape_h_w)  # 修改通道数
        
        # 修改backbone的第一层以接受更多通道
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        
        # 替换第一层卷积以支持多通道输入
        if total_channels != 3:  # 如果不是标准的3通道RGB
            original_conv = backbone_model.conv1
            new_conv = nn.Conv2d(
                total_channels,  # 输入通道数
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # 初始化权重
            with torch.no_grad():
                if total_channels == 3:
                    new_conv.weight.copy_(original_conv.weight)
                else:
                    # 对于多通道，可以复制RGB权重或使用其他初始化策略
                    new_conv.weight[:, :3] = original_conv.weight
                    if total_channels > 3:
                        new_conv.weight[:, 3:] = original_conv.weight[:, :total_channels-3]
            backbone_model.conv1 = new_conv
        
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        # ... 其余代码保持不变 ...
```

### 2. **修改数据处理流程**

在 `DiffusionPolicy` 类中修改图像堆叠方式：

```python
# 在 modeling_diffusion.py 的 DiffusionPolicy 类中
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    # ... 现有代码 ...
    
    if self.config.image_features:
        batch = dict(batch)  # shallow copy
        # 修改这里：将多个摄像头的图像在通道维度上拼接，而不是在批次维度上堆叠
        images_list = [batch[key] for key in self.config.image_features]
        # 在通道维度上拼接 (B, C, H, W) -> (B, C*N, H, W)
        batch[OBS_IMAGES] = torch.cat(images_list, dim=1)  # 在dim=1(通道维度)上拼接
    
    # ... 其余代码保持不变 ...
```

### 3. **修改配置参数**

在策略配置中设置：

```yaml
# 在你的策略配置文件中
use_separate_rgb_encoder_per_camera: false  # 使用共享编码器
# 或者
use_separate_rgb_encoder_per_camera: true   # 使用分离编码器（推荐用于多通道）
```

### 4. **完整的修改示例**

创建一个新的编码器类来处理多通道输入：

```python
# 在 modeling_diffusion.py 中添加
class MultiChannelRgbEncoder(nn.Module):
    """支持多摄像头通道拼接的RGB编码器"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # 计算总通道数
        images_shape = next(iter(config.image_features.values())).shape
        self.num_cameras = len(config.image_features)
        self.channels_per_camera = images_shape[0]  # 通常是3 (RGB)
        self.total_channels = self.channels_per_camera * self.num_cameras
        
        # 设置预处理
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
        
        # 创建backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        
        # 修改第一层以支持多通道输入
        if self.total_channels != 3:
            original_conv = backbone_model.conv1
            new_conv = nn.Conv2d(
                self.total_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # 权重初始化策略
            with torch.no_grad():
                if self.total_channels == 3:
                    new_conv.weight.copy_(original_conv.weight)
                else:
                    # 为每个摄像头复制RGB权重
                    for i in range(self.num_cameras):
                        start_idx = i * self.channels_per_camera
                        end_idx = start_idx + self.channels_per_camera
                        new_conv.weight[start_idx:end_idx] = original_conv.weight
            
            backbone_model.conv1 = new_conv
        
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        
        # 其余部分保持不变
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, self.total_channels, *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """处理多通道拼接的图像输入"""
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x
```

### 5. **修改DiffusionModel以使用新的编码器**

```python
# 在 DiffusionModel.__init__ 中
if self.config.image_features:
    num_images = len(self.config.image_features)
    if self.config.use_separate_rgb_encoder_per_camera:
        # 使用分离编码器（原有逻辑）
        encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
        self.rgb_encoder = nn.ModuleList(encoders)
        global_cond_dim += encoders[0].feature_dim * num_images
    else:
        # 使用多通道编码器
        self.rgb_encoder = MultiChannelRgbEncoder(config)
        global_cond_dim += self.rgb_encoder.feature_dim
```

### 6. **修改图像处理逻辑**

```python
# 在 _prepare_global_conditioning 方法中
if self.config.image_features:
    if self.config.use_separate_rgb_encoder_per_camera:
        # 原有逻辑保持不变
        # ...
    else:
        # 新的多通道处理逻辑
        # 将多个摄像头的图像在通道维度上拼接
        images_list = []
        for i in range(len(self.config.image_features)):
            cam_images = batch["observation.images"][:, :, i]  # (B, S, C, H, W)
            images_list.append(cam_images)
        
        # 在通道维度上拼接
        concatenated_images = torch.cat(images_list, dim=2)  # (B, S, C*N, H, W)
        
        # 重新排列维度以匹配编码器期望的输入格式
        img_features = self.rgb_encoder(
            einops.rearrange(concatenated_images, "b s c h w -> (b s) c h w")
        )
        # 重新排列回原始格式
        img_features = einops.rearrange(
            img_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
        )
        global_cond_feats.append(img_features)
```


[1 tool called]

## 总结

要在训练时增加摄像头的通道维度（而不是简单拼接两个画面），你需要修改以下关键位置：

### **主要修改点：**

1. **`DiffusionRgbEncoder` 类** - 修改以支持多通道输入
2. **`DiffusionModel` 类** - 修改图像处理逻辑
3. **`DiffusionPolicy` 类** - 修改图像堆叠方式
4. **配置参数** - 设置 `use_separate_rgb_encoder_per_camera: false`

### **核心思想：**

- **通道维度拼接**：将多个摄像头的图像在通道维度上拼接 `(B, C, H, W) -> (B, C*N, H, W)`
- **修改backbone**：调整ResNet第一层卷积以接受更多输入通道
- **权重初始化**：为每个摄像头的通道复制预训练权重

### **优势：**

- 模型可以学习不同摄像头之间的空间关系
- 避免了简单拼接可能丢失的空间信息
- 保持了预训练权重的优势

这样修改后，你的模型将能够处理多摄像头输入作为额外的通道维度，而不是简单的画面拼接。