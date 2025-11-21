#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthAnythingV2 FastAPI Server (vitb, torch.load .pth)
"""
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn

from depth_anything_v2.dpt import DepthAnythingV2

# ======================== 配置 ========================
MODEL_ENCODER = "vitb"
DEVICE = "cuda:5"
INPUT_SIZE = 518
PORT = 5555
DEBUG = True
CHECKPOINT_PATH = "/new_data/ff/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"

# vitb 的正确通道配置（与你离线脚本一致）
MODEL_KWARGS = dict(
    encoder="vitb",
    features=128,
    out_channels=[96, 192, 384, 768],
    use_bn=False,
    use_clstoken=False,
)

# ======================== 初始化模型 ========================
print(f"[Server Init] Loading DepthAnythingV2 ({MODEL_ENCODER}) on {DEVICE} ...")
model = DepthAnythingV2(**MODEL_KWARGS)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"[FATAL] checkpoint not found: {CHECKPOINT_PATH}")

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"[Server Warning] Missing keys: {len(missing)}")
if unexpected:
    print(f"[Server Warning] Unexpected keys: {len(unexpected)}")

num_params = sum(p.numel() for p in model.parameters())
print(f"[Server Init] Model loaded with {num_params/1e6:.2f}M parameters on {DEVICE}")

model = model.to(DEVICE).eval()
print(f"[Server Ready] ✅ Loaded checkpoint from {CHECKPOINT_PATH}, listening on 0.0.0.0:{PORT}")

# ======================== FastAPI ========================
app = FastAPI()

@app.post("/infer_depth")
async def infer_depth(file: UploadFile = File(...)):
    """
    接收一张 RGB 图像 (客户端传 BGR)，返回灰度深度 PNG（二进制）
    """
    try:
        # 1) 解码输入
        rgb_bytes = await file.read()
        img_arr = np.frombuffer(rgb_bytes, np.uint8)
        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[ERROR] Received empty/invalid image!")
            return Response(content="Invalid image", status_code=400)

        # 2) BGR->RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 3) 推理 + 稳健归一化
        with torch.no_grad():
            depth = model.infer_image(rgb, INPUT_SIZE)       # float32, HxW
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            dmin, dmax = float(depth.min()), float(depth.max())
            if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin + 1e-8:
                print(f"[WARN] Bad depth range: min={dmin}, max={dmax}. Sending mid-gray.")
                d8 = np.full(depth.shape, 128, dtype=np.uint8)
            else:
                depth = (depth - dmin) / (dmax - dmin + 1e-8)
                d8 = (depth * 255.0).astype(np.uint8)

        # 4) 编码 PNG
        ok, buf = cv2.imencode(".png", d8)
        if not ok:
            print("[ERROR] cv2.imencode failed")
            return Response(content="Encode failed", status_code=500)

        if DEBUG:
            print(f"[Server OK] {datetime.now().isoformat()} shape={d8.shape} range=({d8.min()}, {d8.max()})")

        return Response(content=buf.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"[Server Error] {e}")
        return Response(content=str(e), status_code=500)

# ======================== 启动 ========================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        timeout_keep_alive=30,
        workers=1,
    )
