#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda activate depth
"""
依赖检查
# 虚拟摄像头已就绪就略过；否则：
sudo apt install -y v4l2loopback-utils v4l-utils \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# Python 侧（在 conda env=depth 里）
pip install pyvirtualcam opencv-python torch timm


检查摄像头
ls -l /dev/video30
v4l2-ctl --all -d /dev/video30
调用
ffplay /dev/video30


摄像头伪装

sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback \
  devices=2 \
  video_nr=30,31 \
  card_label="DepthAnything Out A,DepthAnything Out B" \
  exclusive_caps=1



vitl用

CUDA_VISIBLE_DEVICES=0 python3 push_depth.py \
  --in /dev/video0 \
  --out /dev/video30 \
  --width 1280 --height 720 \
  --fps 30 \
  --preview \
  --ckpt checkpoints/depth_anything_v2_vitl.pth \
  --variant vitl \
  --device cuda:0



vitb用

CUDA_VISIBLE_DEVICES=0 python3 push_depth.py \
  --in /dev/video0 \
  --out /dev/video30 \
  --width 1280 --height 720 \
  --fps 30 \
  --ckpt checkpoints/depth_anything_v2_vitb.pth \
  --variant vitb \
  --device cuda:0 \
  --out-format mjpg

CUDA_VISIBLE_DEVICES=0 python3 push_depth.py \
  --in /dev/video6 --out /dev/video30 \
  --width 640 --height 480 --fps 30 \
  --ckpt checkpoints/depth_anything_v2_vitb.pth --variant vitb \
  --device cuda:0 --out-format mjpg
"""

import argparse
import time
import os
from typing import Optional, List

import numpy as np
import cv2
import torch

# ---- pyvirtualcam（优先使用），失败会自动降级到 GStreamer ----
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAS_PYVIRTUALCAM = True
except Exception:
    HAS_PYVIRTUALCAM = False

# ================= 配置 =================

# 你仓库里 DepthAnythingV2 的类路径候选（第一条就是你项目的真实位置）
DA_IMPORT_CANDIDATES: List[str] = [
    "depth_anything_v2.dpt.DepthAnythingV2",
    "depth_anything_v2.model.DepthAnythingV2",
    "depth_anything_v2.models.DepthAnythingV2",
    "depth_anything_v2.DepthAnythingV2",
]

# 输入边长（与官方 demo/训练设定一致会更稳）
DA_INPUT_SIDE = 518  # 常见：384/448/512/518

# ImageNet 规范化（如你仓库不同，请改这里）
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ================= 工具函数 =================

def try_import_depthanything_class() -> type:
    last_err = None
    tried = []
    for path in DA_IMPORT_CANDIDATES:
        try:
            mod_path, cls_name = path.rsplit(".", 1)
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            print(f"[DAv2] import ok: {path}")
            return cls
        except Exception as e:
            tried.append(f"{path}  [fail: {type(e).__name__}: {e}]")
            last_err = e
    msg = "无法导入 DepthAnythingV2 模型类，请按你仓库实际路径调整 DA_IMPORT_CANDIDATES：\n" + "\n".join(tried)
    raise ImportError(msg) from last_err


def open_capture(src: str, width: Optional[int], height: Optional[int], fps: Optional[int]) -> cv2.VideoCapture:
    def _try_open(one):
        cap = cv2.VideoCapture(one, cv2.CAP_V4L2)
        if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:    cap.set(cv2.CAP_PROP_FPS,          fps)
        return cap if cap.isOpened() else None

    cap = _try_open(src)
    if cap: return cap

    try:
        idx = int(src)
        cap = _try_open(idx)
        if cap: return cap
    except Exception:
        pass

    if src == "auto":
        for idx in [0, 1, 2, 3, 4, 5]:
            cap = _try_open(idx)
            if cap:
                print(f"[INFO] auto 选择了摄像头索引: {idx}")
                return cap

    raise RuntimeError(
        f"无法打开输入视频源: {src}\n"
        f"提示：用 `v4l2-ctl --list-devices` / `ls -l /dev/video*` 查可用设备；试试 --in 0/1/2/auto。"
    )


def _open_pyvirtualcam(w: int, h: int, fps: int, out_dev: str, fmt: "PixelFormat"):
    cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, device=out_dev, fmt=fmt)
    print(f"[OK] pyvirtualcam → {out_dev} {w}x{h}@{fps} fmt={fmt.name}")
    return cam

# def _open_gst_writer(w: int, h: int, fps: int, out_dev: str):
#     pipeline = (
#         "appsrc is-live=true block=true format=TIME "
#         "! videoconvert "
#         f"! video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 "
#         "! jpegenc "
#         f"! image/jpeg,framerate={fps}/1 "
#         f"! v4l2sink device={out_dev} sync=false"
#     )
#     writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (w, h))
#     if not writer.isOpened():
#         raise RuntimeError("GStreamer v4l2sink 打不开；请安装 gstreamer1.0-plugins-base/good/bad。")
#     print(f"[OK] GStreamer (MJPEG) → {out_dev} {w}x{h}@{fps}")
#     return writer

def _open_gst_writer(w: int, h: int, fps: int, out_dev: str):
    pipeline = (
        f'appsrc is-live=true block=true format=TIME '
        f'! videoconvert '
        f'! video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 '
        f'! v4l2sink device={out_dev} sync=false'
    )
    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("GStreamer v4l2sink 打不开；请安装 gstreamer1.0-plugins-base/good/bad。")
    print(f"[OK] GStreamer v4l2sink → {out_dev} {w}x{h}@{fps}")
    return writer


# ================= 深度推理：Depth Anything V2 =================

_DA_model = None
_DA_device = None
_DA_variant = None

def _infer_variant_from_ckpt(ckpt_path: str, cli_variant: Optional[str]) -> str:
    if cli_variant:
        v = cli_variant.lower()
        if v in ("vitl", "vitb", "vits", "vitg"):
            return v
        raise ValueError(f"--variant 仅支持 vits/vitb/vitl/vitg，收到：{cli_variant}")
    name = os.path.basename(ckpt_path).lower()
    if "vitb" in name: return "vitb"
    if "vitl" in name: return "vitl"
    if "vits" in name: return "vits"
    if "vitg" in name: return "vitg"
    return "vitl"  # 默认

def load_da_v2_model(ckpt_path: str, device: Optional[str] = None, variant: Optional[str] = None):
    global _DA_model, _DA_device, _DA_variant
    if _DA_model is not None:
        return _DA_model, _DA_device, _DA_variant

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _DA_device = device
    _DA_variant = _infer_variant_from_ckpt(ckpt_path, variant)

    ModelClass = try_import_depthanything_class()

    # 与官方 run.py 一致的配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    cfg = model_configs[_DA_variant]
    model = ModelClass(**cfg)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # 宽松加载，允许头部不匹配（如换 backbone）
    model.load_state_dict(state, strict=False)

    model.eval().to(device)
    _DA_model = model
    print(f"[DAv2] 模型加载完成：variant={_DA_variant}  ckpt={ckpt_path}  device={device}")
    return _DA_model, _DA_device, _DA_variant


@torch.inference_mode()
def _da_preprocess_bchw(frame_bgr: np.ndarray, side: int, device: str) -> torch.Tensor:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scale = side / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    rgb_rs = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_h = max(0, side - nh)
    pad_w = max(0, side - nw)
    top = pad_h // 2; bottom = pad_h - top
    left = pad_w // 2; right = pad_w - left
    rgb_pad = cv2.copyMakeBorder(rgb_rs, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    x = torch.from_numpy(rgb_pad).float().permute(2,0,1).unsqueeze(0) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device, non_blocking=True), (top, left, nh, nw, h, w)

@torch.inference_mode()
def _da_postprocess_to_vis(depth: np.ndarray, w: int, h: int, crop: tuple) -> np.ndarray:
    top, left, nh, nw, H, W = crop
    depth = depth[top:top+nh, left:left+nw]
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC)
    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

@torch.inference_mode()
def run_depthanything(frame_bgr: np.ndarray, ckpt_path: str, device: Optional[str] = None, variant: Optional[str] = None) -> np.ndarray:
    model, device, _ = load_da_v2_model(ckpt_path, device, variant)

    # 优先使用仓库自带 API（若存在）
    if hasattr(model, "infer_image"):
        try:
            depth = model.infer_image(frame_bgr, DA_INPUT_SIDE)  # 期望返回 HxW float
            h, w = frame_bgr.shape[:2]
            return _da_postprocess_to_vis(np.asarray(depth, dtype=np.float32), w, h, (0,0,h,w,h,w))
        except Exception as e:
            print(f"[WARN] infer_image 调用失败，改用通用前向：{e}")

    # 通用路径：自己做预处理 → model(x) → 还原
    h, w = frame_bgr.shape[:2]
    x, crop = _da_preprocess_bchw(frame_bgr, DA_INPUT_SIDE, device)
    pred = model(x)
    if isinstance(pred, (list, tuple)): pred = pred[-1]
    if pred.dim() == 4 and pred.shape[1] == 1: pred = pred.squeeze(1)
    depth = pred.squeeze(0).float().detach().cpu().numpy()
    return _da_postprocess_to_vis(depth, w, h, crop)


# ================= 备用：MiDaS =================

_MIDAS_model = None
_MIDAS_tf = None

@torch.inference_mode()
def run_midas(frame_bgr: np.ndarray) -> np.ndarray:
    global _MIDAS_model, _MIDAS_tf
    if _MIDAS_model is None:
        try:
            _MIDAS_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            _MIDAS_model.eval()
            _MIDAS_tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        except Exception as e:
            raise RuntimeError(
                f"加载 MiDaS 失败：{e}\n"
                f"提示：若缺少 timm，请 `pip install timm`，或不要使用 --midas。"
            )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = _MIDAS_tf(rgb).unsqueeze(0)
    pred = _MIDAS_model(x).squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    depth_u8 = (pred * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


# ================= 主流程 =================

def pipeline(
    in_src: str,
    out_dev: str,
    width: Optional[int],
    height: Optional[int],
    fps: int,
    show_preview: bool,
    use_midas: bool,
    ckpt_path: str,
    device: Optional[str] = None,
    variant: Optional[str] = None,
):
    cap = open_capture(in_src, width, height, fps)

    # 读取首帧确定尺寸
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("首帧读取失败：检查摄像头是否被占用或权限不足")

    if width and height and (frame.shape[1] != width or frame.shape[0] != height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]

    # 选择推理函数
    def infer_fn(img_bgr: np.ndarray) -> np.ndarray:
        if use_midas:
            return run_midas(img_bgr)
        else:
            return run_depthanything(img_bgr, ckpt_path=ckpt_path, device=device, variant=variant)

    # --- 打开输出：优先 pyvirtualcam，失败则降级 GStreamer ---
    fmt = PixelFormat.BGR if HAS_PYVIRTUALCAM else None
    cam = None
    writer = None
    try:
        if HAS_PYVIRTUALCAM:
            cam = _open_pyvirtualcam(w, h, fps, out_dev, fmt)
        else:
            raise RuntimeError("pyvirtualcam 不可用")
    except Exception as e:
        print(f"[WARN] pyvirtualcam 不可用，降级到 GStreamer：{e}")
        writer = _open_gst_writer(w, h, fps, out_dev)

    print(f"[RUN] 推理中，写入 {out_dev} ...  按 ESC 退出")

    n = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            out_bgr = infer_fn(frame)

            if cam is not None:
                cam.send(out_bgr)
                cam.sleep_until_next_frame()
            else:
                writer.write(out_bgr)

            if show_preview:
                cv2.imshow("Depth Preview → /dev/video30", out_bgr)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                    break

            n += 1
            if n % max(1, fps * 5) == 0:
                dt = time.time() - t0
                print(f"[stats] {n} frames in {dt:.1f}s  -> {n/dt:.2f} FPS")
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        if cam is not None:
            cam.close()
        if writer is not None:
            writer.release()


def main():
    ap = argparse.ArgumentParser(description="Camera → DepthAnythingV2(ViT)/MiDaS → /dev/videoX (v4l2loopback)")
    ap.add_argument("--in", dest="in_src", default="/dev/video0", help="输入视频源（/dev/videoX 或 索引或 'auto'）")
    ap.add_argument("--out", dest="out_dev", default="/dev/video30", help="输出虚拟设备路径（默认 /dev/video30）")
    ap.add_argument("--width", type=int, default=None, help="输出宽（默认跟输入首帧一致）")
    ap.add_argument("--height", type=int, default=None, help="输出高（默认跟输入首帧一致）")
    ap.add_argument("--fps", type=int, default=30, help="输出帧率（默认 30）")
    ap.add_argument("--preview", action="store_true", help="打开本地预览窗口")
    ap.add_argument("--midas", action="store_true", help="用 MiDaS 先跑通链路（需 pip install timm）")
    ap.add_argument("--variant", default=None, help="vits/vitb/vitl/vitg（留空=按 ckpt 文件名自动推断）")
    ap.add_argument("--ckpt", dest="ckpt_path", default="checkpoints/depth_anything_v2_vitl.pth",
                    help="DepthAnythingV2 权重路径（文件名里含 vitl/vitb 可自动识别）")
    ap.add_argument("--device", default=None, help="推理设备：cuda / cpu（默认自动）")
    args = ap.parse_args()

    pipeline(
        in_src=args.in_src,
        out_dev=args.out_dev,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_preview=args.preview,
        use_midas=args.midas,
        ckpt_path=args.ckpt_path,
        device=args.device,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
