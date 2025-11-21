#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests
import numpy as np
import cv2

class DepthClient:
    def __init__(self, server_url="http://192.168.3.101:5555/infer_depth", timeout=5.0,
                 debug_save_dir=None):
        """
        server_url: DepthAnything FastAPI 服务器地址
        timeout: 网络超时秒数
        debug_save_dir: 若指定路径，则会在每次推理后保存覆盖的调试图片
        """
        self.server_url = server_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

        self.debug_save_dir = debug_save_dir
        if self.debug_save_dir:
            os.makedirs(self.debug_save_dir, exist_ok=True)

    def __call__(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """输入 BGR 图像，返回灰度深度图 (H,W,1) ∈ [0,1]"""
        try:
            ok, img_encoded = cv2.imencode(".png", rgb_bgr)
            if not ok:
                raise RuntimeError("cv2.imencode() failed — invalid RGB input")

            resp = self.session.post(
                self.server_url,
                files={"file": ("image.png", img_encoded.tobytes(), "image/png")},
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                print(f"[DepthClient][WARN] HTTP {resp.status_code}")
                return np.zeros(rgb_bgr.shape[:2] + (1,), dtype=np.float32)

            else:
                # 未开启调试仍需解码
                depth_u8 = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
                if depth_u8 is None:
                    return np.zeros(rgb_bgr.shape[:2] + (1,), dtype=np.float32)

            # 归一化为 float32 (H,W,1)
            depth = (depth_u8.astype(np.float32) / 255.0)[..., None]
            return depth

        except requests.exceptions.Timeout:
            print(f"[DepthClient][ERROR] Timeout ({self.timeout}s) while contacting {self.server_url}")
        except requests.exceptions.ConnectionError:
            print(f"[DepthClient][ERROR] Cannot connect to DepthAnything server at {self.server_url}")
        except Exception as e:
            print(f"[DepthClient][ERROR] Unexpected error: {e}")

        return np.zeros(rgb_bgr.shape[:2] + (1,), dtype=np.float32)

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass
