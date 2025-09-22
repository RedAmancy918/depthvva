#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 计算相机外参（world=棋盘坐标系）

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import cv2
import yaml


def load_intrinsics(calib_path: str):
    data = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    K = np.array(data["color"]["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["color"]["D"], dtype=np.float64).reshape(-1, 1)
    return K, D


def load_board_yaml(board_yaml: str):
    cfg = yaml.safe_load(Path(board_yaml).read_text(encoding="utf-8"))
    assert cfg.get("type", "checkerboard") == "checkerboard"
    rows = int(cfg["rows"])
    cols = int(cfg["columns"])
    sq = float(cfg["square_size"])
    return rows, cols, sq


def build_object_points(rows: int, cols: int, square: float):
    """
    OpenCV 的 patternSize=(列, 行)；这里生成棋盘平面上的 3D 点 (Z=0)：
    原点在一个角点，X 轴沿列方向，Y 轴沿行方向。
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0] = grid_x.flatten() * square
    objp[:, 1] = grid_y.flatten() * square
    return objp  # (N,3)


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    # 先常规，再 SB 兜底
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        res = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if isinstance(res, tuple):
            corners = res[0]
            ok = corners is not None and len(corners) == pattern_size[0] * pattern_size[1]
    if ok:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return ok, corners


def to_T(Rm: np.ndarray, t: np.ndarray):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t.reshape(3)
    return T


def average_rotation_matrices(Rs: list[np.ndarray]) -> np.ndarray:
    """
    对一组旋转矩阵做“矩阵平均 + SVD 投影到 SO(3)”（不依赖 SciPy）
    参考：对 R 求和后做极分解/正交投影
    """
    if len(Rs) == 1:
        return Rs[0]
    M = np.zeros((3, 3), dtype=np.float64)
    for R in Rs:
        M += R
    M /= float(len(Rs))
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    # 保证是 proper rotation（det=+1）
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1.0
        R_avg = U @ Vt
    return R_avg


def main():
    ap = argparse.ArgumentParser("Calibrate camera extrinsics to checkerboard world")
    ap.add_argument("--images_glob", default="samples/color_*.png", help="输入图像通配符")
    ap.add_argument("--calib", required=True, help="camera_calib.json（color 相机 K/D）")
    ap.add_argument("--board", required=True, help="board.yaml（rows/columns/square_size[米]）")
    ap.add_argument("--out", default="cam_world_extrinsics.json", help="输出 JSON 路径")
    ap.add_argument("--viz_dir", default="viz_calib", help="角点可视化输出目录")
    ap.add_argument("--reproj_outlier_ratio", type=float, default=2.0,
                    help="以中位数的多少倍作为离群阈值，默认 2.0")
    args = ap.parse_args()

    K, D = load_intrinsics(args.calib)
    rows, cols, sq = load_board_yaml(args.board)
    pattern_size = (cols, rows)
    objp = build_object_points(rows, cols, sq)

    img_paths = sorted(glob.glob(args.images_glob))
    assert img_paths, f"No images matched: {args.images_glob}"

    Ts_cam_to_world: list[np.ndarray] = []
    per_view_err: list[tuple[str, float]] = []

    Path(args.viz_dir).mkdir(parents=True, exist_ok=True)

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern_size)
        if not ok:
            print(f"[skip] corners not found: {p}")
            continue

        # solvePnP：求 棋盘->相机
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print(f"[skip] solvePnP failed: {p}")
            continue
        R_tc, _ = cv2.Rodrigues(rvec)
        T_target_to_cam = to_T(R_tc, tvec)
        T_cam_to_target = np.linalg.inv(T_target_to_cam)  # 相机->棋盘(=world)

        # 重投影误差
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        e = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean()
        per_view_err.append((p, float(e)))

        # 可视化保存
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        cv2.imwrite(str(Path(args.viz_dir) / f"{Path(p).stem}_corners.png"), vis)

        Ts_cam_to_world.append(T_cam_to_target)

    assert Ts_cam_to_world, "No valid views."

    # 离群剔除
    errs = np.array([e for _, e in per_view_err], dtype=np.float64)
    med = np.median(errs)
    thr = med * float(args.reproj_outlier_ratio)
    keep_idx = [i for i, (_, e) in enumerate(per_view_err) if e <= thr]

    if len(keep_idx) < len(Ts_cam_to_world):
        print(f"[info] Filtered {len(Ts_cam_to_world) - len(keep_idx)} outliers by reprojection error "
              f"(median={med:.3f}px, thr={thr:.3f}px)")

    # 若有效帧太少或你希望最稳妥，直接选误差最小的一帧
    if len(keep_idx) == 0:
        print("[warn] 所有视图都被判为离群，退化为选择全局最小误差帧。")
        idx_best = int(np.argmin(errs))
        T_cam_to_world = Ts_cam_to_world[idx_best]
        used = [idx_best]
    else:
        # 在保留集合里选一个“最佳帧”（误差最小）
        keep_errs = errs[keep_idx]
        idx_best_in_keep = keep_idx[int(np.argmin(keep_errs))]

        # 也可做多帧融合：旋转矩阵 SVD 平均 + 平移均值
        Rs = [Ts_cam_to_world[i][:3, :3] for i in keep_idx]
        ts = [Ts_cam_to_world[i][:3, 3] for i in keep_idx]
        R_avg = average_rotation_matrices(Rs)
        t_avg = np.mean(np.stack(ts, axis=0), axis=0)

        T_cam_to_world_fused = to_T(R_avg, t_avg)
        # 可选：与“最佳帧”差距过大时，保守起见用最佳帧
        def rot_err_deg(Ra, Rb):
            # 角轴法估计两个旋转的夹角（度）
            Rrel = Ra.T @ Rb
            cos_theta = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        best_R = Ts_cam_to_world[idx_best_in_keep][:3, :3]
        ang = rot_err_deg(best_R, R_avg)
        if ang > 3.0:  # 角度差过大就用最佳帧（阈值可调）
            print(f"[warn] 平均旋转与最佳帧差 {ang:.2f}°，使用最佳帧结果。")
            T_cam_to_world = Ts_cam_to_world[idx_best_in_keep]
            used = [idx_best_in_keep]
        else:
            T_cam_to_world = T_cam_to_world_fused
            used = keep_idx

    T_world_to_cam = np.linalg.inv(T_cam_to_world)

    # 输出
    out = {
        "board": {"type": "checkerboard", "rows": rows, "columns": cols, "square_size_m": sq},
        "intrinsics": {"K": K.tolist(), "D": D.reshape(-1).tolist()},
        "stats": {
            "num_images": len(img_paths),
            "num_used": len(used),
            "reproj_error_mean_px": float(errs.mean()),
            "reproj_error_median_px": float(med),
            "used_images": [per_view_err[i][0] for i in used],
        },
        "T_cam_to_world": T_cam_to_world.tolist(),
        "T_world_to_cam": T_world_to_cam.tolist(),
        "note": "world = checkerboard frame: X along columns, Y along rows, Z out of board."
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    np.set_printoptions(precision=6, suppress=True)
    print("T_cam_to_world =\n", T_cam_to_world)
    print("Saved:", args.out, " | Viz in:", args.viz_dir)


if __name__ == "__main__":
    main()
