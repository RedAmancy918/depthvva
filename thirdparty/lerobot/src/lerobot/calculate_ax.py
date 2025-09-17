#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 计算相机外参
import argparse, glob, json
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial.transform import Rotation as R

def load_intrinsics(calib_path):
    data = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    K = np.array(data["color"]["K"], dtype=np.float64).reshape(3,3)
    D = np.array(data["color"]["D"], dtype=np.float64).reshape(-1,1)
    return K, D

def load_board_yaml(board_yaml):
    cfg = yaml.safe_load(Path(board_yaml).read_text(encoding="utf-8"))
    assert cfg.get("type","checkerboard") == "checkerboard"
    rows = int(cfg["rows"]); cols = int(cfg["columns"]); sq=float(cfg["square_size"])
    return rows, cols, sq

def build_object_points(rows, cols, square):
    # patternSize=(cols, rows)；原点在一个角点，X:列、Y:行、Z=0
    objp = np.zeros((rows*cols, 3), np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:,0] = grid_x.flatten() * square
    objp[:,1] = grid_y.flatten() * square
    return objp  # (N,3)

def find_corners(gray, pattern_size):
    # 先常规，再 SB 兜底
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        res = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if isinstance(res, tuple):
            corners = res[0]
            ok = corners is not None and len(corners)==pattern_size[0]*pattern_size[1]
    if ok:
        term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term)
    return ok, corners

def to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = Rm
    T[:3, 3] = t.reshape(3)
    return T

def main():
    ap = argparse.ArgumentParser("Calibrate camera extrinsics to checkerboard world")
    ap.add_argument("--images_glob", default="samples/color_*.png")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--out", default="cam_world_extrinsics.json")
    ap.add_argument("--viz_dir", default="viz_calib")
    args = ap.parse_args()

    K, D = load_intrinsics(args.calib)
    rows, cols, sq = load_board_yaml(args.board)
    pattern_size = (cols, rows)
    objp = build_object_points(rows, cols, sq)

    img_paths = sorted(glob.glob(args.images_glob))
    assert img_paths, f"No images matched: {args.images_glob}"

    Ts_cam_to_world = []
    per_view_err = []

    Path(args.viz_dir).mkdir(parents=True, exist_ok=True)

    for p in img_paths:
        img = cv2.imread(p); 
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern_size)
        if not ok:
            print(f"[skip] corners not found: {p}")
            continue
        # solvePnP: 求棋盘->相机
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print(f"[skip] solvePnP failed: {p}")
            continue
        R_tc, _ = cv2.Rodrigues(rvec)
        T_target_to_cam = to_T(R_tc, tvec)
        T_cam_to_target = np.linalg.inv(T_target_to_cam)  # 相机->棋盘(=world)

        # 计算重投影误差
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        e = np.linalg.norm(proj.reshape(-1,2)-corners.reshape(-1,2), axis=1).mean()
        per_view_err.append((p, float(e)))

        # 可视化保存
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        cv2.imwrite(str(Path(args.viz_dir)/f"{Path(p).stem}_corners.png"), vis)

        Ts_cam_to_world.append(T_cam_to_target)

    assert Ts_cam_to_world, "No valid views."

    # 鲁棒平均：对旋转用四元数平均，对平移用中位数/均值（剔除大误差帧）
    errs = np.array([e for _,e in per_view_err])
    med = np.median(errs)
    keep = [i for i,(_,e) in enumerate(per_view_err) if e < med*2.0]  # 简单剔除离群
    if len(keep) < len(Ts_cam_to_world):
        print(f"Filtered {len(Ts_cam_to_world)-len(keep)} outliers by reprojection error")

    Rs = []; ts = []
    for i in keep:
        T = Ts_cam_to_world[i]
        Rs.append(R.from_matrix(T[:3,:3]))
        ts.append(T[:3,3])
    # 旋转平均
    R_avg = R.mean(Rs).as_matrix()
    # 平移平均
    t_avg = np.mean(np.stack(ts,0), axis=0)
    T_cam_to_world = to_T(R_avg, t_avg)
    T_world_to_cam = np.linalg.inv(T_cam_to_world)

    out = {
        "board": {"type":"checkerboard","rows":rows,"columns":cols,"square_size_m":sq},
        "intrinsics": {"K": K.tolist(), "D": D.reshape(-1).tolist()},
        "stats": {
            "num_images": len(img_paths),
            "num_used": len(keep),
            "reproj_error_mean_px": float(errs.mean()),
            "reproj_error_median_px": float(med),
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
