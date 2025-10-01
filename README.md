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
训练：
```bash
python /home/ubuntu/code/ff/lerobot/src/lerobot/scripts/train.py \
  --dataset.repo_id=feng_sylvie00/so101_train02 \
  --dataset.root=/home/ubuntu/code/ff/so101_train03_hand_depth \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so101_03_hand_depth \
  --job_name=diffusion_so101_exp_03 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=128 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false
```

推理：
**如果输入是热力图像记得改index**
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

## 2025.9.18
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
- depthanythingv2处理图像会默认同时输出原rgb和热力图 横向拼接在一起 如果只需要热力图的话应该运行run_video_only.py

**run_video_only.py**
```py
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str, required=True, help="Path to a video file or directory")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='output grayscale depth map')
    
    args = parser.parse_args()
    
    # 自动选择设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[INFO] Using device: {DEVICE}")
    
    # 模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 初始化并加载权重
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    state_dict = torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location=DEVICE)
    depth_anything.load_state_dict(state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # 处理输入路径
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        # 输出分辨率与原视频一致
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            # 推理
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # 直接写入深度图像
            out.write(depth)
        
        raw_video.release()
        out.release()
```
**dp+depthany**
1. 主摄 RGB+深度热力 腕部 RGB+深度热力 path:
2. 主摄 深度热力     腕部 深度热力 path:
3. 主摄 RGB         腕部 深度热力 path:

## 2025.09.27
- **测试记录**
- 叠放块 腕部深度热力+fixed rgb 在权重20000的时候效果最好 但是夹取的成功率都不高 考虑是采集数据的时候下爪的角度比较偏
- 对比：腕部+fixed都rgb 
- 腕部热力的启动的快一些 效果还是要好一些的

- **文件名称整理**
- train01
task description: put the cube into circle
- train02
task description: stack two cubes

_depth: 两个摄像头都输入热力图
_depth_hand:fixed 输入普通rgb，handeye输入热力图
- outputs命名规则同上
