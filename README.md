# depthvva
## Day1 2025.9.14
- **把臂的端口名字固定为了 blue_follower、blue_leader**
- **之前连接深度相机的线竟然是usb2.1，换了一根usb3**
- **lerobot可以读取深度信息但是没有办法保存，目前考虑把realsense官方读取保存的代码嵌进lerobot的**
- **之前设置的realsense的帧率是15，所以最后的视频特别快，我就说为什么我上次采的深度数据是动作的一半。现在改回30了。**

数据采集脚本
```bash
python -m lerobot.record   --robot.disable_torque_on_disconnect=true         --robot.type=so101_follower     --robot.port=/dev/blue_follower         --robot.id=R12253310      --robot.cameras="{'handeye':{'type':'opencv','index_or_path':10,'width':640,'height':480,'fps':30},'fixed':{'type':'intelrealsense','serial_number_or_name':'250122073394','width':640,'height':480,'fps':30,'use_depth':true}}"    --teleop.type=so101_leader        --teleop.port=/dev/blue_leader  --teleop.id=R07253310   --display_data=true       --dataset.root="$HOME/datasets/lerobot/so101"   --dataset.push_to_hub=false       --dataset.repo_id="local/so101"         --dataset.num_episodes=10       --dataset.episode_time_s=20       --dataset.single_task='Place the tape'  --dataset.fps=30
```