import argparse
import cv2
import matplotlib
import numpy as np
import os
import time
import torch
from contextlib import nullcontext

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--cam-id', type=int, default=0, help='camera device id')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    parser.add_argument('--no-save', dest='no_save', action='store_true', help='不保存视频文件')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 检查权重文件是否存在
    weight_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 打开摄像头
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开摄像头 {args.cam_id}')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30

    # 设置视频保存
    out = None
    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        output_width = frame_width if args.pred_only else frame_width * 2 + 50
        timestr = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.outdir, f'webcam_{args.cam_id}_{timestr}.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                            frame_rate, (output_width, frame_height))
        print(f'保存到: {output_path}')

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    autocast_ctx = torch.cuda.amp.autocast if DEVICE == 'cuda' else nullcontext

    print(f'使用设备: {DEVICE}')
    print('按 q 退出')

    try:
        with torch.inference_mode():
            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                with autocast_ctx():
                    depth = depth_anything.infer_image(raw_frame, args.input_size)

                # 归一化深度图
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)

                if args.grayscale:
                    depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                # 显示结果
                if args.pred_only:
                    display_frame = depth_vis
                    if out: out.write(depth_vis)
                else:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    display_frame = cv2.hconcat([raw_frame, split_region, depth_vis])
                    if out: out.write(display_frame)

                # 实时显示
                cv2.imshow('Depth Estimation', display_frame)
                
                # 按 q 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()