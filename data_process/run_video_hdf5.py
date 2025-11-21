#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Tuple

import h5py
import numpy as np
import cv2
import torch


# ============== utils ==============

def add_path(p):
    if p and p not in sys.path:
        sys.path.insert(0, p)

def decode_rgb(frame) -> np.ndarray:
    if isinstance(frame, (bytes, np.void)):
        arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed: not a valid image buffer")
        return img
    elif isinstance(frame, np.ndarray):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Unexpected array shape for RGB: {frame.shape}")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")

def open_rgb_dataset(root: h5py.File, camera: str):
    ds_path = f"/observation/{camera}/rgb"
    if ds_path not in root:
        raise KeyError(f"Dataset not found: {ds_path}")
    return root[ds_path]

def infer_depth_model(encoder: str, device: str, input_size: int, da_path: str, ckpt_override: str = None):
    add_path(da_path)
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder: {encoder}")

    model = DepthAnythingV2(**model_configs[encoder])
    ckpt = ckpt_override or os.path.join(da_path, "checkpoints", f"depth_anything_v2_{encoder}.pth")
    if not os.path.exists(ckpt):
        alt = ckpt.replace(".pth", ".safetensors")
        if os.path.exists(alt):
            ckpt = alt
        else:
            raise FileNotFoundError(f"[FATAL] Checkpoint not found: {ckpt}")

    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt)
        model.load_state_dict(state)
    else:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state)

    model = model.to(device).eval()

    def infer_bgr_uint8(bgr_img: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            depth = model.infer_image(bgr_img, input_size)
        dmin, dmax = float(depth.min()), float(depth.max())
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            raise RuntimeError(f"Invalid depth range: min={dmin}, max={dmax}")
        depth = (depth - dmin) / (dmax - dmin)
        d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        return d8

    return infer_bgr_uint8, encoder, device, os.path.abspath(ckpt)

def ensure_depth_dataset(group_ob_cam: h5py.Group, name: str, shape: Tuple[int, int, int], overwrite: bool):
    if name in group_ob_cam:
        if overwrite:
            del group_ob_cam[name]
        else:
            return group_ob_cam[name], False
    ds = group_ob_cam.create_dataset(
        name, shape=shape, dtype='uint8',
        chunks=(min(shape[0], 64), shape[1], shape[2]),
        compression="gzip", compression_opts=4
    )
    return ds, True


# ============== main ==============

def process_single_h5(h5_path, args, infer_fn, enc_used, dev_used, ckpt_used):
    print(f"\n[Processing] {h5_path}")
    t0 = time.time()
    used_frames = 0
    with h5py.File(h5_path, "r+") as f:
        rgb_ds = open_rgb_dataset(f, args.camera)
        cam_group = f[f"/observation/{args.camera}"]

        if (rgb_ds.dtype == np.uint8 and getattr(rgb_ds, "ndim", None) == 4 and rgb_ds.shape[-1] == 3):
            T, H, W, _ = rgb_ds.shape
        else:
            img0 = decode_rgb(rgb_ds[0])
            H, W = img0.shape[:2]
            T = rgb_ds.shape[0]

        depth_ds, created = ensure_depth_dataset(cam_group, args.dataset_name, (T, H, W), args.overwrite)
        if not created:
            print(f"[Skip] Dataset already exists: /observation/{args.camera}/{args.dataset_name}")
            return

        print(f"[Info] Writing depth to {h5_path} shape=({T},{H},{W})")

        for i in range(T):
            bgr = decode_rgb(rgb_ds[i])
            d8 = infer_fn(bgr)
            depth_ds[i, :, :] = d8
            used_frames += 1
            if (i + 1) % 50 == 0 or i == T - 1:
                print(f"  progress: {i + 1}/{T}", end="\r")
        print()

        ts = datetime.now().isoformat()
        cam_group.attrs.update({
            "depthanything.strict": True,
            "depthanything.encoder": enc_used,
            "depthanything.device": dev_used,
            "depthanything.ckpt": ckpt_used,
            "depthanything.input_size": int(args.input_size),
            "depthanything.dataset_name": args.dataset_name,
            "depthanything.timestamp": ts,
            "depthanything.frames": int(used_frames),
        })
        depth_ds.attrs.update({
            "source": "depth-anything",
            "encoder": enc_used,
            "device": dev_used,
            "ckpt": ckpt_used,
            "timestamp": ts,
            "dtype_note": "uint8 [0..255]",
        })

    print(f"[Done] Wrote {used_frames}/{T} frames. Time: {time.time() - t0:.2f}s")


def main():
    ap = argparse.ArgumentParser("Add depth_gray into HDF5 from RGB using DepthAnythingV2 (supports directory)")
    ap.add_argument("--h5", required=True, help="Path to episode*.hdf5 or directory containing them")
    ap.add_argument("--camera", default="head_camera")
    ap.add_argument("--da-path", required=True)
    ap.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt")
    ap.add_argument("--dataset-name", default="depth_gray")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"[DA] STRICT mode on. encoder={args.encoder} device={args.device}")
    infer_fn, enc_used, dev_used, ckpt_used = infer_depth_model(
        args.encoder, args.device, args.input_size, args.da_path, args.ckpt
    )
    print(f"[DA] Model loaded. ckpt={ckpt_used}")

    # 🔹 支持目录输入：遍历其中的所有 .hdf5 文件
    if os.path.isdir(args.h5):
        h5_files = sorted(
            [os.path.join(args.h5, f) for f in os.listdir(args.h5) if f.endswith(".hdf5")]
        )
        if not h5_files:
            print(f"[Error] No .hdf5 files found in directory: {args.h5}")
            return
        for f in h5_files:
            process_single_h5(f, args, infer_fn, enc_used, dev_used, ckpt_used)
    else:
        process_single_h5(args.h5, args, infer_fn, enc_used, dev_used, ckpt_used)


if __name__ == "__main__":
    main()
