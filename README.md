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