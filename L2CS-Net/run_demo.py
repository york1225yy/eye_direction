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


def _gaze_direction_label(pitch_deg: float, yaw_deg: float) -> str:
    """Convert pitch/yaw angles to a human-readable gaze direction string."""
    v_thr, h_thr = 10.0, 10.0  # degree thresholds
    v = "Up" if pitch_deg > v_thr else ("Down" if pitch_deg < -v_thr else "Center")
    h = "Right" if yaw_deg > h_thr else ("Left" if yaw_deg < -h_thr else "Center")
    if v == "Center" and h == "Center":
        return "Forward"
    if v == "Center":
        return h
    if h == "Center":
        return v
    return f"{v}-{h}"


def _draw_semi_bg(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  color=(0, 0, 0), alpha: float = 0.55):
    """Draw a semi-transparent filled rectangle."""
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_info_overlay(frame: np.ndarray, results, fps: float, frame_idx: int = 0):
    """Draw all gaze pipeline results onto the frame."""
    h_frame, w_frame = frame.shape[:2]

    # ── Top-left: global stats ────────────────────────────────────────────
    n_faces = len(results.pitch) if results is not None else 0
    top_lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {n_faces}",
        f"Frame: {frame_idx}",
    ]
    line_h = 24
    panel_w = 160
    panel_h = line_h * len(top_lines) + 8
    _draw_semi_bg(frame, 4, 4, panel_w, panel_h)
    for li, txt in enumerate(top_lines):
        cy = 4 + 6 + li * line_h + line_h - 4
        if li == 0:
            color = (0, 255, 0)
            scale, thick = 0.7, 2
        else:
            color = (220, 220, 220)
            scale, thick = 0.55, 1
        cv2.putText(frame, txt, (8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if results is None or n_faces == 0:
        return frame

    # ── Per-face annotation ───────────────────────────────────────────────
    for i in range(n_faces):
        pitch_rad = float(results.pitch[i])
        yaw_rad   = float(results.yaw[i])
        pitch_deg = pitch_rad * 180.0 / np.pi
        yaw_deg   = yaw_rad   * 180.0 / np.pi
        score     = float(results.scores[i])
        bbox      = results.bboxes[i]

        x_min = max(int(bbox[0]), 0)
        y_min = max(int(bbox[1]), 0)
        x_max = min(int(bbox[2]), w_frame)
        y_max = min(int(bbox[3]), h_frame)
        bw    = x_max - x_min
        bh    = y_max - y_min

        direction = _gaze_direction_label(pitch_deg, yaw_deg)

        # ── Info lines for this face ──────────────────────────────────────
        info_lines = [
            (f"Face #{i+1}",              (0, 255, 255)),
            (f"Score:  {score:.3f}",      (200, 200, 200)),
            (f"Pitch:  {pitch_deg:+.2f} deg", _angle_color(pitch_deg)),
            (f"Yaw:    {yaw_deg:+.2f} deg",   _angle_color(yaw_deg)),
            (f"Dir:    {direction}",       (255, 200, 50)),
            (f"BBox:   {bw}x{bh} px",     (180, 180, 180)),
            (f"  @ ({x_min},{y_min})",     (140, 140, 140)),
        ]

        il_h    = 20
        il_w    = 190
        total_h = il_h * len(info_lines) + 8

        # place panel below bbox if space allows, otherwise above
        px = x_min
        if y_max + total_h + 4 < h_frame:
            py = y_max + 2
        else:
            py = max(y_min - total_h - 2, 0)

        _draw_semi_bg(frame, px, py, il_w, total_h, color=(20, 20, 20), alpha=0.65)

        for li, (txt, col) in enumerate(info_lines):
            cy = py + 6 + li * il_h + il_h - 4
            cv2.putText(frame, txt, (px + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        # ── Landmarks (5 points from RetinaFace) ─────────────────────────
        if results.landmarks is not None and results.landmarks.shape[0] > i:
            lms = results.landmarks[i]  # shape (5, 2): [leye, reye, nose, lmouth, rmouth]
            lm_labels = ["LE", "RE", "N", "LM", "RM"]
            lm_colors = [
                (255, 100, 100),   # left eye  – blue-ish
                (100, 100, 255),   # right eye – red-ish
                (100, 255, 100),   # nose      – green
                (255, 255, 100),   # left mouth
                (255, 100, 255),   # right mouth
            ]
            for li_idx, (lx, ly) in enumerate(lms):
                cx, cy = int(lx), int(ly)
                if 0 <= cx < w_frame and 0 <= cy < h_frame:
                    cv2.circle(frame, (cx, cy), 4, lm_colors[li_idx], -1, cv2.LINE_AA)
                    cv2.putText(frame, lm_labels[li_idx],
                                (cx + 5, cy - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                lm_colors[li_idx], 1, cv2.LINE_AA)

    return frame


def _angle_color(deg: float):
    """Green near 0°, yellow at ±20°, red beyond ±40°."""
    abs_deg = abs(deg)
    if abs_deg < 20:
        return (80, 255, 80)
    if abs_deg < 40:
        return (0, 200, 255)
    return (0, 80, 255)


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
                frame = draw_info_overlay(frame, results, fps_display, frame_idx)
            else:
                # no face detected – draw global stats only
                frame = draw_info_overlay(frame, results, fps_display, frame_idx)

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
