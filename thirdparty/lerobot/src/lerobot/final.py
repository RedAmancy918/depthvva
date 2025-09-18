#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense D435：按固定间隔采集彩色 PNG，并导出相机内参与外参。

启动后会输出：
- d435_intrinsics.json       # 包含 color/depth 内参、depth_scale、depth->color 外参等
- camera_calib.json          # 仅含 color K、D（给外参标定脚本直接使用）
- images/000000.png ...      # 彩色图片（BGR 存盘，cv2读写）

用法示例：
python final.py \
  --output ./out_rs \
  --width 1280 --height 720 --fps 30 \
  --interval 2.0 \
  --max-count 20 \
  --fix-exposure --exposure-us 200 --wb-k 4500
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    print("未找到 pyrealsense2，请先安装: pip install pyrealsense2", file=sys.stderr)
    raise

# 可选图像保存库
_HAS_CV2 = False
_HAS_IMAGEIO = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    pass

try:
    import imageio.v2 as imageio  # type: ignore
    _HAS_IMAGEIO = True
except Exception:
    pass


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: Tuple[float, float, float, float, float]


def save_image_bgr_png(path: str, bgr: np.ndarray) -> None:
    """保存BGR图像为PNG。优先cv2，其次imageio（自动转RGB），兜底npy。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if _HAS_CV2:
        ok = cv2.imwrite(path, bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite 失败: {path}")
        return
    if _HAS_IMAGEIO:
        imageio.imwrite(path, bgr[:, :, ::-1])  # imageio 期望 RGB
        return
    np.save(os.path.splitext(path)[0] + ".npy", bgr)


def intrinsics_from_profile(vsp: rs.video_stream_profile) -> CameraIntrinsics:
    intr = vsp.get_intrinsics()
    return CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        model=str(intr.model),
        coeffs=(
            float(intr.coeffs[0]),
            float(intr.coeffs[1]),
            float(intr.coeffs[2]),
            float(intr.coeffs[3]),
            float(intr.coeffs[4]),
        ),
    )


def _get_depth_scale(profile: rs.pipeline_profile) -> float:
    """
    更鲁棒的 depth_scale 获取方式：
    - 优先使用 first_depth_sensor() / depth_sensor.get_depth_scale()
    - 其次把通用 sensor 强转为 depth_sensor 再取
    - 兜底从 option.depth_units 读取（单位米/单位）
    """
    dev = profile.get_device()

    # 方案1：专用 API（部分版本提供）
    try:
        ds = dev.first_depth_sensor()  # 若不可用会抛异常
        return float(ds.get_depth_scale())
    except Exception:
        pass

    # 方案2：遍历传感器，强转为 depth_sensor
    for s in dev.sensors:
        try:
            ds = rs.depth_sensor(s)  # 等价于 as_depth_sensor
            return float(ds.get_depth_scale())
        except Exception:
            continue

    # 方案3：读取 depth_units 选项
    for s in dev.sensors:
        try:
            if s.supports(rs.option.depth_units):
                return float(s.get_option(rs.option.depth_units))
        except Exception:
            continue

    raise RuntimeError("未能获取 depth_scale：未找到 DepthSensor 或不支持 depth_units 选项")


def export_full_intrinsics(profile: rs.pipeline_profile, out_json: str) -> None:
    """导出全量内参/外参 + camera_calib.json（color K/D）。"""
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

    color_intr = intrinsics_from_profile(color_profile)
    depth_intr = intrinsics_from_profile(depth_profile)

    # 深度->彩色 外参
    ext = depth_profile.get_extrinsics_to(color_profile)
    R = [float(x) for x in ext.rotation]      # 9元素，行主序
    t = [float(x) for x in ext.translation]   # 单位：米

    depth_scale = _get_depth_scale(profile)

    dev = profile.get_device()

    def _get(info: int, default: str = "unknown") -> str:
        return dev.get_info(info) if dev.supports(info) else default

    meta = {
        "device_name": _get(rs.camera_info.name),
        "serial": _get(rs.camera_info.serial_number),
        "product_line": _get(rs.camera_info.product_line),
        "stream": {
            "resolution": {
                "color": [color_intr.width, color_intr.height],
                "depth": [depth_intr.width, depth_intr.height],
            },
        },
        "color_intrinsics": asdict(color_intr),
        "depth_intrinsics": asdict(depth_intr),
        "depth_to_color_extrinsics": {
            "R_3x3_row_major": R,
            "t_xyz_m": t,
        },
        "depth_scale_m_per_unit": depth_scale,
        "note": "R为行主序9元素，可重排为3x3；t单位米。",
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 再写一份精简版（color相机 K/D），给外参标定脚本直接用
    K = [
        color_intr.fx, 0.0, color_intr.ppx,
        0.0, color_intr.fy, color_intr.ppy,
        0.0, 0.0, 1.0,
    ]
    D = list(color_intr.coeffs)
    with open(os.path.join(os.path.dirname(out_json), "camera_calib.json"), "w", encoding="utf-8") as f:
        json.dump({"color": {"K": K, "D": D}}, f, ensure_ascii=False, indent=2)


def set_manual_exposure_white_balance(profile: rs.pipeline_profile,
                                      exposure_us: int = 200,
                                      white_balance_k: int = 4500) -> None:
    """（可选）固定曝光和白平衡，提升标定角点稳定性。"""
    try:
        dev = profile.get_device()
        # 找到支持曝光/白平衡选项的传感器（通常是 RGB）
        color_sensor = None
        for s in dev.sensors:
            if s.supports(rs.option.enable_auto_exposure) and s.supports(rs.option.exposure):
                color_sensor = s
                break
        if color_sensor is None:
            return

        # 关闭自动曝光 & 设置手动曝光
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if color_sensor.supports(rs.option.exposure):
            color_sensor.set_option(rs.option.exposure, float(exposure_us))

        # 关闭自动白平衡并设置固定值（并非所有RGB传感器都支持）
        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        if color_sensor.supports(rs.option.white_balance):
            color_sensor.set_option(rs.option.white_balance, float(white_balance_k))
    except Exception as e:
        print(f"[warn] 设置手动曝光/白平衡失败：{e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RealSense D435：导出内参并按间隔采集彩色图片")
    p.add_argument("--output", required=True, help="输出目录：将生成 d435_intrinsics.json / camera_calib.json / images/")
    p.add_argument("--width", type=int, default=1280, help="彩色流宽度，默认1280")
    p.add_argument("--height", type=int, default=720, help="彩色流高度，默认720")
    p.add_argument("--fps", type=int, default=30, help="帧率，默认30")
    p.add_argument("--depth-width", type=int, default=None, help="深度流宽度（不填则与彩色相同）")
    p.add_argument("--depth-height", type=int, default=None, help="深度流高度（不填则与彩色相同）")
    p.add_argument("--interval", type=float, default=2.0, help="保存间隔秒，默认2.0")
    p.add_argument("--max-count", type=int, default=None, help="最多保存多少张；不填为无限直到停止")
    p.add_argument("--fix-exposure", action="store_true", help="固定曝光/白平衡（更稳的角点检测，标定推荐开启）")
    p.add_argument("--exposure-us", type=int, default=200, help="曝光时间（微秒），--fix-exposure 时有效")
    p.add_argument("--wb-k", type=int, default=4500, help="白平衡开尔文温度，--fix-exposure 时有效")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.output)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    depth_w = args.depth_width if args.depth_width else args.width
    depth_h = args.depth_height if args.depth_height else args.height

    pipeline = rs.pipeline()
    config = rs.config()
    # 同时启用彩色与深度（深度是为了导出完整内参/外参）
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, args.fps)

    profile: Optional[rs.pipeline_profile] = None
    try:
        profile = pipeline.start(config)

        # 预热，保证自动曝光稳定（若后续要切手动，先让图像稳定一下也无妨）
        for _ in range(5):
            pipeline.wait_for_frames()

        if args.fix_exposure:
            set_manual_exposure_white_balance(profile, exposure_us=args.exposure_us, white_balance_k=args.wb_k)

        # 导出内参/外参
        intr_path = os.path.join(out_dir, "d435_intrinsics.json")
        export_full_intrinsics(profile, intr_path)
        print(f"[info] 已导出 D435 内参/外参 -> {intr_path}")
        print(f"[info] 同时生成 camera_calib.json（color K/D）供外参标定使用。")

        saved = 0
        last_save_time = 0.0
        print("[info] 开始采集：按 Ctrl+C 停止。")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            now = time.time()
            if (now - last_save_time) >= float(args.interval):
                color_bgr = np.asanyarray(color_frame.get_data())
                fname = f"{saved:06d}.png"
                save_path = os.path.join(images_dir, fname)
                save_image_bgr_png(save_path, color_bgr)
                print(f"保存: {save_path}")

                saved += 1
                last_save_time = now
                if args.max_count is not None and saved >= int(args.max_count):
                    break

    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        print("已停止相机。")


if __name__ == "__main__":
    main()
