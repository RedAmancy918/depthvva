#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel RealSense D435 点云捕获工具
用于从Intel RealSense D435相机捕获深度数据并保存为PLY格式的点云文件
支持彩色纹理和深度对齐功能
"""

import os
import time
import argparse
import pyrealsense2 as rs

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    ap = argparse.ArgumentParser(
        description="从Intel RealSense D435捕获点云并保存为.ply文件"
    )
    # 输出目录参数
    ap.add_argument("-o", "--out_dir", type=str, default="./pointclouds",
                    help="保存.ply文件的目录")
    # 捕获帧数参数
    ap.add_argument("-n", "--num", type=int, default=1,
                    help="要保存的帧数 (1 = 单次拍摄)")
    # 流帧率参数
    ap.add_argument("--fps", type=int, default=30, help="流帧率")
    # 流宽度参数
    ap.add_argument("--width", type=int, default=640, help="流宽度")
    # 流高度参数
    ap.add_argument("--height", type=int, default=480, help="流高度")
    # 是否禁用颜色参数
    ap.add_argument("--no_color", action="store_true",
                    help="不为点云使用颜色纹理")
    # 对齐方式参数
    ap.add_argument("--align_to", choices=["color", "depth"], default="color",
                    help="在生成点云之前将帧集对齐到'color'或'depth'")
    return ap.parse_args()

def main():
    """
    主函数：配置RealSense相机，捕获点云数据并保存为PLY文件
    """
    # 解析命令行参数
    args = parse_args()
    # 创建输出目录（如果不存在）
    os.makedirs(args.out_dir, exist_ok=True)

    # 配置RealSense流
    pipeline = rs.pipeline()  # 创建管道对象
    config = rs.config()      # 创建配置对象
    
    # 启用深度流：宽度x高度，Z16格式，指定帧率
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    
    # 如果未禁用颜色，则启用彩色流：BGR8格式
    if not args.no_color:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    # 启动流管道
    profile = pipeline.start(config)

    # 可选：设置深度传感器的高精度预设（可选，可调整）
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        # 2 = 高精度, 3 = 高密度 (数值可能因固件版本而异)
        depth_sensor.set_option(rs.option.visual_preset, 2)
    except Exception:
        pass  # 不关键，忽略错误

    # 配置帧对齐
    # 根据参数选择对齐到彩色流或深度流
    align_to = rs.stream.color if (args.align_to == "color" and not args.no_color) else rs.stream.depth
    align = rs.align(align_to)  # 创建对齐对象

    # 创建点云计算器
    pc = rs.pointcloud()

    # 相机预热：丢弃前几帧以确保稳定
    print("[INFO] 正在预热相机...")
    for _ in range(5):
        pipeline.wait_for_frames()

    saved = 0  # 已保存的文件计数
    try:
        # 主捕获循环
        while saved < args.num:
            # 等待并获取帧
            frames = pipeline.wait_for_frames()
            # 对齐帧（将深度和彩色帧对齐到同一坐标系）
            frames = align.process(frames)

            # 获取深度帧
            depth = frames.get_depth_frame()
            if not depth:
                continue  # 如果没有深度帧，跳过

            # 获取彩色帧（如果启用）
            color = None
            if not args.no_color:
                color = frames.get_color_frame()

            # 将深度映射到彩色（如果彩色可用）
            if color:
                pc.map_to(color)  # 将点云映射到彩色帧
            
            # 计算点云
            points = pc.calculate(depth)  # 返回rs.points对象

            # 保存为PLY文件
            ts_ms = int(time.time() * 1000)  # 时间戳（毫秒）
            out_path = os.path.join(args.out_dir, f"d435_{ts_ms}.ply")
            # 导出为PLY格式：filename, textured_frame（纹理帧可以为None）
            points.export_to_ply(out_path, color if color else depth)

            # 可选：嵌入法线标志（如果提供颜色，PLY将包含顶点+颜色）
            # 这里没有直接的法线；可以在点云工具中稍后估算

            # 输出保存信息
            total = points.size()
            print(f"[SAVED] {out_path}  (点数: {total}, 颜色: {'是' if color else '否'})")
            saved += 1

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断操作。")
    finally:
        # 停止管道并清理资源
        pipeline.stop()
        print("[INFO] 管道已停止。")

if __name__ == "__main__":
    # 程序入口点
    main()
