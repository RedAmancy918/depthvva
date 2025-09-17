#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#获取正运动学矩阵    
"""
实用工具：计算并保存当前正运动学得到的 T_base->ee 4x4 齐次变换矩阵

依赖：`lerobot.model.kinematics.RobotKinematics`（需要安装可选依赖 placo）

用法示例：
  1) 直接传关节角（单位：度）
     python cam_arm.py \
        --urdf /path/to/robot.urdf \
        --joints 0,15,-30,45,10,0 \
        --target_frame gripper_frame_link \
        --out T_base_ee.json

  2) 指定关节名顺序（可选，不指定则使用URDF顺序）
     python cam_arm.py \
        --urdf /path/to/robot.urdf \
        --joint_names joint1,joint2,joint3,joint4,joint5,joint6 \
        --joints 0,15,-30,45,10,0

  3) 输出为 .npy：
     python cam_arm.py --urdf /path.urdf --joints 0,0,0,0,0,0 --out T.npy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional

import numpy as np

from lerobot.model.kinematics import RobotKinematics


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def compute_fk_matrix(
    urdf_path: str,
    joint_deg: Iterable[float],
    target_frame_name: str = "gripper_frame_link",
    joint_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    计算正运动学，返回 T_base->ee (4x4)
    Args:
        urdf_path: 机器人URDF路径
        joint_deg: 关节角（度）序列，长度应与关节数匹配
        target_frame_name: 末端执行器在URDF中的frame名
        joint_names: 关节名顺序（可选）。不提供则使用URDF默认顺序
    Returns:
        4x4 numpy.ndarray 齐次变换矩阵
    """
    kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=joint_names,
    )
    joint_array = np.asarray(list(joint_deg), dtype=float)
    T = kin.forward_kinematics(joint_array)
    return T


def save_transform(T: np.ndarray, out_path: str) -> None:
    """
    保存 4x4 变换矩阵到文件：
      - .json/.txt 保存为列表（嵌套list）
      - .npy 保存为 numpy 二进制
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".json", ".txt"):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(T.tolist(), f, ensure_ascii=False, indent=2)
    elif ext == ".npy":
        np.save(out_path, T)
    else:
        # 默认按json保存
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(T.tolist(), f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="计算并保存 T_base->ee (FK)"
    )
    ap.add_argument("--urdf", required=True, type=str, help="URDF 文件路径")
    ap.add_argument("--joints", required=True, type=str, help="逗号分隔的关节角(度)，例如: 0,15,-30,45,10,0")
    ap.add_argument("--joint_names", type=str, default=None, help="逗号分隔的关节名顺序，可选")
    ap.add_argument("--target_frame", type=str, default="gripper_frame_link", help="末端执行器frame名")
    ap.add_argument("--out", type=str, default="T_base_ee.json", help="输出文件路径（.json/.txt/.npy）")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    joint_vals = parse_float_list(args.joints)
    joint_names = parse_str_list(args.joint_names) if args.joint_names else None

    T = compute_fk_matrix(
        urdf_path=args.urdf,
        joint_deg=joint_vals,
        target_frame_name=args.target_frame,
        joint_names=joint_names,
    )

    save_transform(T, args.out)
    # 控制台友好打印
    np.set_printoptions(precision=6, suppress=True)
    print("T_base->ee =")
    print(T)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()


