#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    """_summary_

    Returns:
        _type_: _description_
    """
import os
import time
import argparse
import pyrealsense2 as rs

def parse_args():
    ap = argparse.ArgumentParser(
        description="Capture point clouds from Intel RealSense D435 and save as .ply"
    )
    ap.add_argument("-o", "--out_dir", type=str, default="./pointclouds",
                    help="Directory to save .ply files")
    ap.add_argument("-n", "--num", type=int, default=1,
                    help="Number of frames to save (1 = single shot)")
    ap.add_argument("--fps", type=int, default=30, help="Stream FPS")
    ap.add_argument("--width", type=int, default=640, help="Stream width")
    ap.add_argument("--height", type=int, default=480, help="Stream height")
    ap.add_argument("--no_color", action="store_true",
                    help="Do not use color texture for the point cloud")
    ap.add_argument("--align_to", choices=["color", "depth"], default="color",
                    help="Align framesets to 'color' or 'depth' before generating point cloud")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if not args.no_color:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    # Start streaming
    profile = pipeline.start(config)

    # Optional: set high accuracy preset for depth (optional, can be tuned)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        # 2 = High Accuracy, 3 = High Density (values may vary by FW)
        depth_sensor.set_option(rs.option.visual_preset, 2)
    except Exception:
        pass  # not critical

    # Align frames
    align_to = rs.stream.color if (args.align_to == "color" and not args.no_color) else rs.stream.depth
    align = rs.align(align_to)

    # Pointcloud calculator
    pc = rs.pointcloud()

    print("[INFO] Warming up the camera...")
    for _ in range(5):
        pipeline.wait_for_frames()

    saved = 0
    try:
        while saved < args.num:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth = frames.get_depth_frame()
            if not depth:
                continue

            # Optional color for texturing
            color = None
            if not args.no_color:
                color = frames.get_color_frame()

            # Map depth to color (if color available)
            if color:
                pc.map_to(color)
            points = pc.calculate(depth)  # rs.points

            # Save as PLY
            ts_ms = int(time.time() * 1000)
            out_path = os.path.join(args.out_dir, f"d435_{ts_ms}.ply")
            # export_to_ply(filename, textured_frame) — textured_frame can be None
            points.export_to_ply(out_path, color if color else depth)

            # Optionally embed normals flag (PLY will contain vertices+colors if color provided)
            # No direct normals here; you can estimate later in point cloud tools.

            total = points.size()
            print(f"[SAVED] {out_path}  (points: {total}, color: {'yes' if color else 'no'})")
            saved += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pipeline.stop()
        print("[INFO] Pipeline stopped.")

if __name__ == "__main__":
    main()
