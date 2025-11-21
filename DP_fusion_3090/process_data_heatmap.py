#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Streaming version: pack RoboTwin HDF5 -> Zarr
兼容所有 zarr 版本，不会 OOM。
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import zarr
import shutil


def load_hdf5(dataset_path):
    """读取单个 HDF5 episode."""
    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        vector = root["/joint_action/vector"][()]

        # RGB：按相机名组织；通常我们用 head_camera
        image_dict = {}
        for cam_name in root["/observation"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

        depth_path = "/observation/head_camera/depth_gray"
        if depth_path not in root:
            raise KeyError(f"Missing {depth_path}")
        depth_gray = root[depth_path][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict, depth_gray


def decode_rgb_any(x):
    """支持 bytes 或 (H,W,3)uint8 → BGR uint8"""
    if isinstance(x, (bytes, np.void)):
        arr = np.frombuffer(x, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return img
    elif isinstance(x, np.ndarray):
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Unexpected RGB shape: {x.shape}")
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def main():
    parser = argparse.ArgumentParser(description="Pack RoboTwin episodes safely to Zarr (streaming mode)")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    load_dir = os.path.join("../../data", task_name, task_config)
    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print(f"[Info] Streaming pack: {load_dir} → {save_dir}")
    root = zarr.open_group(save_dir, mode="w")
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    episode_ends = []
    total_count = 0

    for ep in range(num):
        h5_path = os.path.join(load_dir, f"data/episode{ep}.hdf5")
        if not os.path.exists(h5_path):
            print(f"⚠️ Missing {h5_path}, skip.")
            continue
        print(f"  → Processing episode {ep}: {h5_path}")

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
            depth_gray_all,
        ) = load_hdf5(h5_path)

        rgb_seq = image_dict_all["head_camera"]
        T = len(rgb_seq)

        rgb_frames, depth_frames, states, actions = [], [], [], []
        for j in range(T):
            if j != T - 1:
                rgb = decode_rgb_any(rgb_seq[j])
                depth = depth_gray_all[j]
                rgb_frames.append(rgb)
                depth_frames.append(depth)
                states.append(vector_all[j])
            if j != 0:
                actions.append(vector_all[j])

        rgb_np = np.moveaxis(np.array(rgb_frames, dtype=np.uint8), -1, 1)  # (N,3,H,W)
        depth_np = np.expand_dims(np.array(depth_frames, dtype=np.uint8), 1)  # (N,1,H,W)
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32)

        N = rgb_np.shape[0]
        if ep == 0:
            # 初始化 dataset（可扩展）
            H, W = rgb_np.shape[2:]
            D = states_np.shape[1]
            data_grp.create_dataset(
                "head_camera",
                shape=(0, 3, H, W),
                chunks=(10, 3, H, W),
                dtype="uint8",
                compressor=compressor,
            )
            data_grp.create_dataset(
                "head_camera_depth",
                shape=(0, 1, H, W),
                chunks=(10, 1, H, W),
                dtype="uint8",
                compressor=compressor,
            )
            data_grp.create_dataset(
                "state",
                shape=(0, D),
                chunks=(100, D),
                dtype="float32",
                compressor=compressor,
            )
            data_grp.create_dataset(
                "action",
                shape=(0, D),
                chunks=(100, D),
                dtype="float32",
                compressor=compressor,
            )

        # --- ✅ 兼容 zarr v2/v3 resize 写法 ---
        for key, arr in zip(
            ["head_camera", "head_camera_depth", "state", "action"],
            [rgb_np, depth_np, states_np, actions_np],
        ):
            ds = data_grp[key]
            old_shape = ds.shape
            new_shape = (old_shape[0] + arr.shape[0],) + old_shape[1:]
            ds.resize(new_shape)
            ds[old_shape[0] : new_shape[0]] = arr

        total_count += N
        episode_ends.append(total_count)

    # --- Meta ---
    meta_grp.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64))
    print("\n✅ Packing finished successfully.")
    print(f"Total samples: {total_count}, Episodes: {len(episode_ends)}")

    print("\n📦 Zarr structure check:")
    print("Root keys:", list(root.keys()))
    print("data/:", list(data_grp.keys()))
    print("meta/:", list(meta_grp.keys()))


if __name__ == "__main__":
    main()
