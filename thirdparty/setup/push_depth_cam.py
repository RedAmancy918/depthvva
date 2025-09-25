#!/usr/bin/env python3

"""
将相机视频流转换为深度热力图并发布为虚拟摄像头。

功能概述（精简版）：
- 读取视频源（/dev/videoX 或编号）
- 使用 Depth-Anything-V2（指定 encoder 与 checkpoint）进行单目深度估计
- 将深度映射到彩色热力图
- 通过 pyvirtualcam 发布为 v4l2loopback 虚拟摄像头

依赖：
- OpenCV: pip install opencv-python
- PyTorch: pip install torch torchvision
- pyvirtualcam: pip install pyvirtualcam（需要系统安装 v4l2loopback）
- Depth-Anything-V2 仓库与对应 checkpoint

示例：
python thirdparty/setup/push_depth_cam.py \
  --source /dev/video6 \
  --fps 30 --width 1280 --height 720 \
  --vcam_device /dev/video10 \
  --da2_repo /home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2 \
  --da2_checkpoint /home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth \
  --encoder vitb --input_size 448 \
  --device cuda
  
run:
python thirdparty/setup/push_depth_cam.py \
  --source 0 \
  --publish vcam --vcam_device /dev/video10 \
  --width 1280 --height 720 --fps 30 \
  --da2_repo /home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2 \
  --da2_checkpoint /home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth \
  --encoder vitl --input_size 518 \
  --device auto --model auto

说明：需要系统加载 v4l2loopback 且安装 pyvirtualcam。
"""

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np
import os
import torch
import pyvirtualcam


def select_device(name: str) -> str:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if name in {"cuda", "cpu"}:
        return name
    return "cpu"


class DepthEstimator:
    def __init__(self, device: str = "auto",
                 da2_repo: Optional[str] = None, da2_checkpoint: Optional[str] = None,
                 encoder: str = "vitb", input_size: int = 448,
                 disable_xformers: bool = True):
        self.device_str = select_device(device)
        self.model = None
        self.transform = None
        self.input_size = int(input_size)
        self.disable_xformers = disable_xformers

        # 仅加载本地 Depth-Anything-V2（必须指定仓库与权重）
        if not (da2_repo and da2_checkpoint):
            raise RuntimeError("必须提供 --da2_repo 与 --da2_checkpoint")
        if self.disable_xformers:
            os.environ["XFORMERS_DISABLED"] = "1"
        if os.path.isdir(da2_repo) and da2_repo not in sys.path:
            sys.path.insert(0, da2_repo)
        # 在导入 DA2 之前，准备强制替换 xformers 注意力为 PyTorch SDPA
        # 这样无论 CPU 还是 CUDA 都走稳定路径，避免 NotImplementedError
        def _install_attention_fallback():
            try:
                import depth_anything_v2.dinov2_layers.attention as attn_mod  # type: ignore
                import torch as _torch
                def _fallback_mea(q, k, v, attn_bias=None, p: float = 0.0):
                    return _torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=p, is_causal=False
                    )
                attn_mod.memory_efficient_attention = _fallback_mea  # type: ignore
            except Exception:
                pass

        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        _install_attention_fallback()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        enc = encoder if encoder in model_configs else 'vitb'
        model = DepthAnythingV2(**model_configs[enc])
        state = torch.load(da2_checkpoint, map_location='cpu')
        model.load_state_dict(state)
        self.model = model

        # 迁移到设备
        self.device = torch.device(self.device_str)
        self.model.to(self.device)

    @staticmethod
    def _normalize_to_01(depth: np.ndarray, clip_low_q: float = 2.0, clip_high_q: float = 98.0) -> np.ndarray:
        # 百分位裁剪，增强稳定性
        d = depth.astype(np.float32)
        lo = np.percentile(d, clip_low_q)
        hi = np.percentile(d, clip_high_q)
        if hi <= lo:
            hi = d.max() if d.max() > 0 else 1.0
            lo = d.min()
        d = np.clip(d, lo, hi)
        d = (d - lo) / (hi - lo + 1e-8)
        return d

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        with np.errstate(all="ignore"):
            depth = self.model.infer_image(bgr, self.input_size)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return depth.astype(np.float32)

    def depth_to_colormap(self, depth: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
        d01 = self._normalize_to_01(depth)
        d255 = (d01 * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(d255, cmap)
        return colored  # BGR


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    # source 可以是整型摄像头编号、文件路径或URL
    cap: Optional[cv2.VideoCapture] = None
    # 设备文件优先尝试 V4L2 后端
    if not source.isdigit() and source.startswith("/dev/video"):
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        except Exception:
            cap = None
    # 回退普通打开
    if cap is None or not cap.isOpened():
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

    # 基本属性设置
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    # 有些UVC相机需要设置MJPG四字符码以达到更高分辨率/帧率
    try:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc == 0:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    return cap


def main():
    parser = argparse.ArgumentParser(description="Push depth heatmap video via virtual camera")
    parser.add_argument("--source", type=str, default="0", help="视频源：/dev/videoX 或编号，默认0")
    parser.add_argument("--width", type=int, default=640, help="输入/输出宽度")
    parser.add_argument("--height", type=int, default=480, help="输入/输出高度")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    parser.add_argument("--device", type=str, default="auto", help="计算设备：auto/cuda/cpu")
    parser.add_argument("--vcam_device", type=str, default="/dev/video10", help="虚拟摄像头设备（v4l2loopback）")
    parser.add_argument("--cmap", type=str, default="turbo", choices=["turbo", "jet", "inferno", "magma", "plasma"], help="颜色映射")
    # Depth-Anything-V2 相关参数
    parser.add_argument("--da2_repo", type=str, default="/home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2", help="Depth-Anything-V2 仓库路径")
    parser.add_argument("--da2_checkpoint", type=str, default="/home/rb01/Geo/depthvva/thirdparty/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth", help="DA2 权重路径")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl", "vitg"], help="DA2 编码器选择")
    parser.add_argument("--input_size", type=int, default=448, help="DA2 推理输入尺寸")
    parser.add_argument("--disable_xformers", action="store_true", default=True, help="禁用 xformers 以避免不兼容运算符（默认启用禁用）")
    args = parser.parse_args()

    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    cmap = cmap_map.get(args.cmap, cv2.COLORMAP_TURBO)

    # 打开视频源
    cap = open_capture(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"无法打开视频源: {args.source}")
        sys.exit(1)

    # 加载深度估计
    print("加载深度模型中……")
    estimator = DepthEstimator(
        device=args.device,
        da2_repo=args.da2_repo,
        da2_checkpoint=args.da2_checkpoint,
        encoder=args.encoder,
        input_size=args.input_size,
        disable_xformers=args.disable_xformers,
    )
    print(f"设备: {estimator.device_str}")

    # 发布器
    writer = None
    cam = None
    try:
        # 使用 RGB 输出，避免帧大小不匹配（RGB 需要 H*W*3）。
        cam = pyvirtualcam.Camera(
            device=args.vcam_device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
        )
        print(f"已连接虚拟摄像头: {args.vcam_device}")
    except Exception as e:
        raise RuntimeError(f"虚拟摄像头不可用: {e}")

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取帧失败，结束。")
                break

            # 调整尺寸到指定输出大小
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            # 深度推理与上色
            depth = estimator.infer(frame)
            heat = estimator.depth_to_colormap(depth, cmap=cmap)  # BGR

            # 推送到虚拟摄像头（RGB）
            cam.send(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()

            # 简单的FPS指示
            now = time.time()
            if now - prev_time >= 2.0:
                prev_time = now
    finally:
        cap.release()
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


