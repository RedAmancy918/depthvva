#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import shutil
from copy import deepcopy

import numpy as np
import cv2
import h5py
import zarr
import yaml  # 若无依赖可删

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm     = root["/joint_action/left_arm"][()]
        right_gripper= root["/joint_action/right_gripper"][()]
        right_arm    = root["/joint_action/right_arm"][()]
        vector       = root["/joint_action/vector"][()]

        # RGB：按相机名组织；通常我们用 head_camera
        image_dict = {}
        for cam_name in root["/observation"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]  # (T,) bytes 或 (T,H,W,3) uint8

        # 深度（灰度）：只取 head_camera 的 depth_gray，必须存在
        depth_path = "/observation/head_camera/depth_gray"
        if depth_path not in root:
            raise KeyError(f"Missing required dataset: {depth_path}")
        depth_gray = root[depth_path][()]  # (T,H,W) uint8

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict, depth_gray

def decode_rgb_any(x):
    """支持 (H,W,3)uint8 (假设RGB) 或 bytes(JPEG/PNG)，返回 BGR uint8。"""
    if isinstance(x, (bytes, np.void)):
        arr = np.frombuffer(x, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("cv2.imdecode failed: invalid image buffer")
        return img
    elif isinstance(x, np.ndarray):
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Unexpected RGB array shape: {x.shape}")
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported RGB frame type: {type(x)}")

def main():
    parser = argparse.ArgumentParser(description="Pack RoboTwin episodes to zarr with RGB + depth_gray")
    parser.add_argument("task_name", type=str, help="Task name (e.g., stack_blocks_two)")
    parser.add_argument("task_config", type=str, help="Config name (e.g., demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)")
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = os.path.join("../../data", task_name, task_config)  # .../<task>/ <config>/
    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays = []
    head_camera_depth_arrays = []  # 新增：深度
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []

    total_count = 0
    current_ep = 0

    print(f"[Info] packing from {load_dir} -> {save_dir}")
    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        h5_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
            depth_gray_all,   # (T,H,W) uint8
        ) = load_hdf5(h5_path)

        # 只用 head_camera
        if "head_camera" not in image_dict_all:
            raise KeyError("head_camera not found under /observation/*/rgb")

        rgb_seq = image_dict_all["head_camera"]  # (T,) bytes 或 (T,H,W,3)
        T = left_gripper_all.shape[0]
        if depth_gray_all.shape[0] != T:
            raise ValueError(f"Depth length mismatch: depth T={depth_gray_all.shape[0]} vs joint T={T} in {h5_path}")

        # 按你原逻辑：图像/状态用 j != (T-1)，动作用 j != 0
        for j in range(T):
            if j != T - 1:
                # 处理 RGB
                head_img = decode_rgb_any(rgb_seq[j])      # (H,W,3) BGR
                head_camera_arrays.append(head_img)

                # 处理 depth_gray 对齐同一帧
                depth_img = depth_gray_all[j]              # (H,W) uint8
                if depth_img.ndim != 2:
                    raise ValueError(f"depth_gray frame shape should be (H,W), got {depth_img.shape}")
                head_camera_depth_arrays.append(depth_img)

                # 状态
                state_arrays.append(vector_all[j])

            if j != 0:
                # 动作：用下一步状态（你原始逻辑是把 joint_state 作为 joint_action 保存）
                joint_action_arrays.append(vector_all[j])

        current_ep += 1
        total_count += (T - 1)   # 为 head_cam/depth/state 累加
        episode_ends_arrays.append(total_count)

    print()  # 换行

    # --- 转 NumPy / 维度整理 ---
    episode_ends_arrays = np.array(episode_ends_arrays, dtype=np.int64)
    state_arrays = np.array(state_arrays, dtype=np.float32)
    joint_action_arrays = np.array(joint_action_arrays, dtype=np.float32)

    head_camera_arrays = np.array(head_camera_arrays, dtype=np.uint8)            # (N,H,W,3) BGR
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)                  # -> (N,3,H,W)

    head_camera_depth_arrays = np.array(head_camera_depth_arrays, dtype=np.uint8)  # (N,H,W)
    head_camera_depth_arrays = head_camera_depth_arrays[:, np.newaxis, ...]        # -> (N,1,H,W)

    # --- 写 zarr ---
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # 计算 chunks
    state_chunk_size = (min(100, state_arrays.shape[0]), state_arrays.shape[1])
    joint_chunk_size = (min(100, joint_action_arrays.shape[0]), joint_action_arrays.shape[1])
    head_camera_chunk = (min(100, head_camera_arrays.shape[0]), *head_camera_arrays.shape[1:])
    head_depth_chunk  = (min(100, head_camera_depth_arrays.shape[0]), *head_camera_depth_arrays.shape[1:])

    # RGB
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk,
        dtype="uint8",
        overwrite=True,
        compressor=compressor,
    )
    # 深度（灰度）
    zarr_data.create_dataset(
        "head_camera_depth",
        data=head_camera_depth_arrays,
        chunks=head_depth_chunk,
        dtype="uint8",
        overwrite=True,
        compressor=compressor,
    )
    # 状态/动作
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    # episode_ends
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    # 可选：标注来源，便于日后追踪
    zarr_data["head_camera_depth"].attrs["source"] = "hdf5:/observation/head_camera/depth_gray"
    zarr_data["head_camera_depth"].attrs["note"] = "uint8 [0..255], N=∑(T-1)"
    print(f"[Done] zarr saved to: {save_dir}")

if __name__ == "__main__":
    main()
