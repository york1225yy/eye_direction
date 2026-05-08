"""
L2CS-Net Gaze Estimation Demo
==============================
Features:
  - Input: webcam (cam) or video file
  - Device: cpu or gpu (cuda)
  - Output: annotated video saved to file (no display required)
  - Overlays: bounding box, gaze arrow, pitch/yaw angles, FPS

Usage examples:
  # Video file, CPU
  python run_demo.py --input datasets/result_long.mp4 --device cpu --output output/result.mp4

  # Webcam, GPU 0
  python run_demo.py --input cam --device gpu:0 --output output/webcam_result.mp4

  # Show window while processing (if display is available)
  python run_demo.py --input datasets/result_long.mp4 --device cpu --output output/result.mp4 --show
"""

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# ── make sure the package is importable even without pip install ──────────────
ROOT = pathlib.Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2cs import Pipeline, render, select_device


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="L2CS-Net Gaze Estimation Demo",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        default="cam",
        help=(
            "Input source.\n"
            "  'cam'       : default webcam (device 0)\n"
            "  'cam:N'     : webcam device N\n"
            "  '<path>'    : path to a video file\n"
            "Default: cam"
        )
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        help=(
            "Inference device.\n"
            "  'cpu'       : CPU\n"
            "  'gpu:0'     : GPU 0  (or just '0')\n"
            "Default: cpu"
        )
    )
    parser.add_argument(
        "--arch",
        default="ResNet50",
        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"],
        help="Backbone architecture. Default: ResNet50"
    )
    parser.add_argument(
        "--weights", "-w",
        default=None,
        help="Path to model weights (.pkl). Default: models/L2CSNet_gaze360.pkl"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/gaze_result.mp4",
        help="Path of output video file. Default: output/gaze_result.mp4"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Face detection confidence threshold. Default: 0.5"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result window in real-time (requires display). Default: off"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the output video. Default: off"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to process (0 = unlimited). Default: 0"
    )
    return parser.parse_args()


def open_source(input_arg: str):
    """Return (cv2.VideoCapture, is_cam: bool)"""
    if input_arg.lower() == "cam":
        cap = cv2.VideoCapture(0)
        return cap, True
    if input_arg.lower().startswith("cam:"):
        cam_id = int(input_arg.split(":")[1])
        cap = cv2.VideoCapture(cam_id)
        return cap, True
    # treat as file path
    path = pathlib.Path(input_arg)
    if not path.exists():
        print(f"[ERROR] Input file not found: {path}")
        sys.exit(1)
    cap = cv2.VideoCapture(str(path))
    return cap, False


def make_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """Create a VideoWriter with the same resolution/fps as the source."""
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    # mp4v codec is widely supported without extra dependencies
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for {out_path}")
        sys.exit(1)
    print(f"[INFO] Output: {out_path}  ({width}x{height} @ {fps:.1f} fps)")
    return writer


def draw_info_overlay(frame: np.ndarray, results, fps: float):
    """Draw FPS counter and pitch/yaw text for each detected face."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2, cv2.LINE_AA
    )
    for i in range(len(results.pitch)):
        pitch_deg = float(results.pitch[i]) * 180.0 / np.pi
        yaw_deg   = float(results.yaw[i])   * 180.0 / np.pi
        bbox      = results.bboxes[i]
        x_min = max(int(bbox[0]), 0)
        y_min = max(int(bbox[1]), 0)

        label = f"P:{pitch_deg:+.1f}  Y:{yaw_deg:+.1f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # background rectangle for readability
        cv2.rectangle(
            frame,
            (x_min, y_min - th - 8),
            (x_min + tw + 4, y_min),
            (0, 0, 0), cv2.FILLED
        )
        cv2.putText(
            frame, label,
            (x_min + 2, y_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Model weights ────────────────────────────────────────────────────────
    if args.weights:
        weights_path = pathlib.Path(args.weights)
    else:
        weights_path = ROOT / "models" / "L2CSNet_gaze360.pkl"

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Run:  python download_model.py")
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = args.device.lower()
    # convert 'gpu:0' -> '0' for select_device
    if device_str.startswith("gpu:"):
        device_str = device_str[4:]
    elif device_str == "gpu":
        device_str = "0"
    # 'cpu' stays as 'cpu'

    cudnn.enabled = True
    device = select_device(device_str, batch_size=1)
    print(f"[INFO] Device: {device}")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {weights_path}")
    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch=args.arch,
        device=device,
        confidence_threshold=args.confidence,
    )
    print("[INFO] Model loaded.")

    # ── Input source ─────────────────────────────────────────────────────────
    cap, is_cam = open_source(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
    source_label = "webcam" if is_cam else args.input
    print(f"[INFO] Input: {source_label}")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")

    # ── Output writer ────────────────────────────────────────────────────────
    writer = None
    if not args.no_save:
        writer = make_writer(args.output, cap)

    # ── Try display ──────────────────────────────────────────────────────────
    show = args.show
    if show:
        # a quick test to see whether a window can actually be created
        try:
            cv2.namedWindow("L2CS-Net Gaze", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[WARN] No display environment detected, disabling --show")
            show = False

    # ── Processing loop ──────────────────────────────────────────────────────
    frame_idx   = 0
    fps_display = 0.0
    t_start     = time.time()

    print("[INFO] Processing... Press 'q' to quit (if --show is enabled).")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_cam:
                    print("[WARN] Failed to read from camera, retrying...")
                    time.sleep(0.05)
                    continue
                else:
                    print("[INFO] End of video.")
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                print(f"[INFO] Reached max-frames limit ({args.max_frames}).")
                break

            t_frame = time.time()

            # ── Inference ────────────────────────────────────────────────
            try:
                results = gaze_pipeline.step(frame)
            except Exception as e:
                print(f"[WARN] Frame {frame_idx}: inference error: {e}")
                results = None

            fps_display = 1.0 / max(time.time() - t_frame, 1e-6)

            # ── Visualise ────────────────────────────────────────────────
            if results is not None and len(results.pitch) > 0:
                frame = render(frame, results)
                frame = draw_info_overlay(frame, results, fps_display)
            else:
                # no face detected – still draw FPS
                cv2.putText(
                    frame, f"FPS: {fps_display:.1f}  No face",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 255), 2, cv2.LINE_AA
                )

            # ── Write ────────────────────────────────────────────────────
            if writer is not None:
                writer.write(frame)

            # ── Display ──────────────────────────────────────────────────
            if show:
                cv2.imshow("L2CS-Net Gaze", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] 'q' pressed, stopping.")
                    break

            # ── Progress log ─────────────────────────────────────────────
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                faces = len(results.pitch) if results is not None else 0
                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    print(
                        f"[INFO] Frame {frame_idx}/{total_frames} ({pct:.1f}%)  "
                        f"FPS: {fps_display:.1f}  Faces: {faces}"
                    )
                else:
                    print(
                        f"[INFO] Frame {frame_idx}  "
                        f"FPS: {fps_display:.1f}  Faces: {faces}"
                    )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps    = frame_idx / max(total_time, 1e-6)
    print(f"\n[DONE] Processed {frame_idx} frames in {total_time:.1f}s (avg {avg_fps:.1f} FPS)")
    if writer is not None:
        out_path = pathlib.Path(args.output)
        size_mb  = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
        print(f"[DONE] Output saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
