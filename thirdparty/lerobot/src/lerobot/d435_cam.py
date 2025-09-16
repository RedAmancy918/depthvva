import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover
    print("未找到 pyrealsense2，请先安装: pip install pyrealsense2", file=sys.stderr)
    raise


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


@dataclass
class RecordingMeta:
    device_name: str
    serial: str
    product_line: str
    depth_scale_m_per_unit: float
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    aligned_to: str  # "color"
    stream_resolution: Tuple[int, int]
    fps: int


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame):  # noqa: ARG002
        self.kill_now = True


class RealSenseRecorder:
    def __init__(
        self,
        output_dir: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        save_color: bool = False,
        record_bag_path: Optional[str] = None,
        align_to_color: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.save_color = save_color
        self.record_bag_path = record_bag_path
        self.align_to_color = align_to_color

        os.makedirs(self.output_dir, exist_ok=True)
        self.depth_dir = os.path.join(self.output_dir, "depth")
        self.color_dir = os.path.join(self.output_dir, "color")
        os.makedirs(self.depth_dir, exist_ok=True)
        if self.save_color:
            os.makedirs(self.color_dir, exist_ok=True)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile: Optional[rs.pipeline_profile] = None
        self.align = rs.align(rs.stream.color) if self.align_to_color else None

        # Time log
        self.timestamps_path = os.path.join(self.output_dir, "timestamps.csv")
        with open(self.timestamps_path, "w", encoding="utf-8") as f:
            f.write("frame_idx,rs_timestamp_ms,system_timestamp_s\n")

        self.meta_path = os.path.join(self.output_dir, "meta.json")

    @staticmethod
    def _intrinsics_from_profile(vsp: rs.video_stream_profile) -> CameraIntrinsics:
        intr = vsp.get_intrinsics()
        return CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            ppx=intr.ppx,
            ppy=intr.ppy,
            model=str(intr.model),
            coeffs=tuple(intr.coeffs),
        )

    def _write_meta(self) -> None:
        if self.profile is None:
            return
        dev = self.profile.get_device()
        name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "unknown"
        serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "unknown"
        product_line = dev.get_info(rs.camera_info.product_line) if dev.supports(rs.camera_info.product_line) else "unknown"

        # Depth scale
        depth_sensor = None
        for s in dev.sensors:
            if s.is_depth_sensor():
                depth_sensor = s
                break
        if depth_sensor is None:
            raise RuntimeError("未找到深度传感器")
        depth_scale = depth_sensor.get_depth_scale()

        # Intrinsics
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        color_intr = self._intrinsics_from_profile(color_profile)
        depth_intr = self._intrinsics_from_profile(depth_profile)

        meta = RecordingMeta(
            device_name=name,
            serial=serial,
            product_line=product_line,
            depth_scale_m_per_unit=float(depth_scale),
            color_intrinsics=color_intr,
            depth_intrinsics=depth_intr,
            aligned_to="color" if self.align_to_color else "none",
            stream_resolution=(self.width, self.height),
            fps=self.fps,
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    def start(self) -> None:
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.record_bag_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.record_bag_path)), exist_ok=True)
            self.config.enable_record_to_file(self.record_bag_path)
        self.profile = self.pipeline.start(self.config)
        # Warm-up few frames for auto-exposure
        for _ in range(5):
            self.pipeline.wait_for_frames()
        self._write_meta()

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _save_frame(
        self,
        frame_idx: int,
        depth_image: np.ndarray,
        color_image: Optional[np.ndarray],
        rs_timestamp_ms: float,
        system_ts_s: float,
    ) -> None:
        # Save depth as .npy (uint16) to preserve raw units
        depth_path = os.path.join(self.depth_dir, f"{frame_idx:06d}.npy")
        np.save(depth_path, depth_image)
        if self.save_color and color_image is not None:
            # Save color as .npy to avoid extra dependency
            color_path = os.path.join(self.color_dir, f"{frame_idx:06d}.npy")
            np.save(color_path, color_image)
        with open(self.timestamps_path, "a", encoding="utf-8") as f:
            f.write(f"{frame_idx},{rs_timestamp_ms:.3f},{system_ts_s:.6f}\n")

    def record(self, max_frames: Optional[int] = None, max_seconds: Optional[float] = None) -> None:
        if self.profile is None:
            self.start()
        killer = GracefulKiller()
        start_time = time.time()
        frame_idx = 0
        try:
            while True:
                if killer.kill_now:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    break

                frames = self.pipeline.wait_for_frames()
                if self.align is not None:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                rs_ts_ms = float(depth_frame.get_timestamp())  # milliseconds
                system_ts_s = time.time()

                depth_image = np.asanyarray(depth_frame.get_data())  # uint16
                color_image = np.asanyarray(color_frame.get_data()) if self.save_color else None

                self._save_frame(frame_idx, depth_image, color_image, rs_ts_ms, system_ts_s)
                frame_idx += 1
        finally:
            self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制 Intel RealSense 深度数据（对齐、带时间戳、可保存）")
    parser.add_argument("--output", required=True, help="输出目录，将保存 depth/、(可选)color/、timestamps.csv、meta.json")
    parser.add_argument("--width", type=int, default=640, help="流宽度，默认 640")
    parser.add_argument("--height", type=int, default=480, help="流高度，默认 480")
    parser.add_argument("--fps", type=int, default=30, help="帧率，默认 30")
    parser.add_argument("--save-color", action="store_true", help="是否同时保存彩色帧（.npy）")
    parser.add_argument("--bag", type=str, default=None, help="若提供路径，则同时录制 .bag")
    parser.add_argument("--max-frames", type=int, default=None, help="最大录制帧数；不填表示无限直到手动停止")
    parser.add_argument("--max-seconds", type=float, default=None, help="最长录制秒数；不填表示无限直到手动停止")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recorder = RealSenseRecorder(
        output_dir=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        save_color=args.save_color,
        record_bag_path=args.bag,
        align_to_color=True,
    )
    print("启动 RealSense 录制... 按 Ctrl+C 停止。")
    recorder.record(max_frames=args.max_frames, max_seconds=args.max_seconds)
    print(f"录制结束，输出目录: {args.output}")


if __name__ == "__main__":
    main()


