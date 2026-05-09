"""
L2CS-Net Gaze Estimation Demo
==============================
Features:
  - Input: webcam (cam) or video file          # 支持摄像头或视频文件输入
  - Device: cpu or gpu (cuda)                  # 支持 CPU / GPU 推理
  - Output: annotated video saved to file      # 输出带标注的视频文件（无需显示器）
  - Overlays: bounding box, gaze arrow, pitch/yaw angles, FPS  # 叠加人脸框、注视箭头、角度、帧率

Usage examples:
  # Video file, CPU
  python run_demo.py --input datasets/result_long.mp4 --device cpu --output output/result.mp4

  # Webcam, GPU 0
  python run_demo.py --input cam --device gpu:0 --output output/webcam_result.mp4

  # Show window while processing (if display is available)
  python run_demo.py --input datasets/result_long.mp4 --device cpu --output output/result.mp4 --show
"""

import argparse           # 用于解析命令行参数
import pathlib            # 跨平台路径操作
import sys                # 系统相关操作（退出、路径）
import time               # 时间计算（FPS、计时）

import cv2                # OpenCV：视频读写、图像绘制
import numpy as np        # 数值计算（数组操作）
import torch              # PyTorch 深度学习框架
import torch.backends.cudnn as cudnn  # cuDNN 加速后端

# ── 确保即使未执行 pip install，也能直接导入 l2cs 包 ──────────────────────────
ROOT = pathlib.Path(__file__).parent.resolve()  # 当前脚本所在目录（绝对路径）
if str(ROOT) not in sys.path:                   # 若该路径不在 Python 搜索路径中
    sys.path.insert(0, str(ROOT))               # 插入到搜索路径最前面，优先导入本地包

from l2cs import Pipeline, render, select_device  # 导入 L2CS 推理管线、渲染函数、设备选择工具


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """解析并返回命令行参数。"""
    parser = argparse.ArgumentParser(
        description="L2CS-Net Gaze Estimation Demo",     # 程序描述
        formatter_class=argparse.RawTextHelpFormatter    # 保留帮助文本原始换行格式
    )
    parser.add_argument(
        "--input", "-i",
        default="cam",       # 默认使用默认摄像头
        help=(
            "Input source.\n"
            "  'cam'       : default webcam (device 0)\n"   # 默认摄像头（设备0）
            "  'cam:N'     : webcam device N\n"             # 指定摄像头编号 N
            "  '<path>'    : path to a video file\n"        # 视频文件路径
            "Default: cam"
        )
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",       # 默认使用 CPU 推理
        help=(
            "Inference device.\n"
            "  'cpu'       : CPU\n"       # CPU 模式
            "  'gpu:0'     : GPU 0  (or just '0')\n"  # GPU 模式（指定卡号）
            "Default: cpu"
        )
    )
    parser.add_argument(
        "--arch",
        default="ResNet50",  # 默认使用 ResNet50 骨干网络
        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"],  # 可选骨干网络
        help="Backbone architecture. Default: ResNet50"
    )
    parser.add_argument(
        "--weights", "-w",
        default=None,        # 不指定时自动查找默认路径
        help="Path to model weights (.pkl). Default: models/L2CSNet_gaze360.pkl"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/gaze_result.mp4",  # 默认输出路径
        help="Path of output video file. Default: output/gaze_result.mp4"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.9,         # 人脸检测置信度阈值，越高误检越少
        help=(
            "Face detection confidence threshold (0~1).\n"
            "  Higher value -> fewer false positives.\n"          # 值越高，误检越少
            "  Recommended: 0.9 for single-driver scene, 0.5 for crowd.\n"  # 单驾驶员场景推荐 0.9
            "Default: 0.9"
        )
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,           # 默认每帧只保留得分最高的 1 张人脸
        help=(
            "Keep only the top N faces (by detection score) per frame.\n"
            "  Use 1 for single-driver monitoring (suppresses false positives).\n"  # 单人监控用 1
            "  Use 0 to keep all detected faces.\n"   # 0 表示保留所有检测到的人脸
            "Default: 1"
        )
    )
    parser.add_argument(
        "--det-scale",
        type=float,
        default=0.5,         # 检测前将帧缩小到原始尺寸的 50%，提升速度
        help=(
            "Scale factor applied to the frame BEFORE face detection (0.1~1.0).\n"
            "  Lower value -> faster detection, reduced accuracy on small faces.\n"   # 值越小越快，但小脸准确率下降
            "  0.5 halves width/height -> ~4x fewer pixels for detector.\n"          # 0.5 使像素量减少约 4 倍
            "  Gaze estimation always uses the original-resolution crop.\n"          # 注视估计始终用原始分辨率裁剪
            "Default: 0.5"
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",  # 开关参数：指定后实时显示结果窗口
        help="Show result window in real-time (requires display). Default: off"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",  # 开关参数：指定后跳过保存输出视频
        help="Skip saving the output video. Default: off"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,            # 0 表示不限制帧数，处理全部帧
        help="Maximum number of frames to process (0 = unlimited). Default: 0"
    )
    return parser.parse_args()  # 解析并返回参数对象


def open_source(input_arg: str):
    """根据输入参数打开视频源，返回 (cv2.VideoCapture, is_cam: bool)。"""
    if input_arg.lower() == "cam":              # 输入为 'cam'，使用默认摄像头（设备0）
        cap = cv2.VideoCapture(0)
        return cap, True                        # is_cam=True 表示实时流
    if input_arg.lower().startswith("cam:"):   # 输入为 'cam:N'，使用第 N 个摄像头
        cam_id = int(input_arg.split(":")[1])  # 解析摄像头编号
        cap = cv2.VideoCapture(cam_id)
        return cap, True
    # 其余情况视为文件路径
    path = pathlib.Path(input_arg)
    if not path.exists():                       # 文件不存在时报错退出
        print(f"[ERROR] Input file not found: {path}")
        sys.exit(1)
    cap = cv2.VideoCapture(str(path))           # 打开视频文件
    return cap, False                           # is_cam=False 表示文件源


def make_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    """创建与源视频分辨率/帧率相同的 VideoWriter 输出对象。"""
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出目录（不存在时）

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 获取源视频宽度（像素）
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取源视频高度（像素）
    fps    = cap.get(cv2.CAP_PROP_FPS)                # 获取源视频帧率
    if fps <= 0:                                       # 帧率无效时（如摄像头）使用默认值
        fps = 25.0

    # mp4v 编解码器无需额外依赖，兼容性好
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")           # 设置编码格式为 mp4v
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))  # 创建写入器
    if not writer.isOpened():                          # 创建失败时报错退出
        print(f"[ERROR] Cannot open VideoWriter for {out_path}")
        sys.exit(1)
    print(f"[INFO] Output: {out_path}  ({width}x{height} @ {fps:.1f} fps)")  # 打印输出信息
    return writer


def _gaze_direction_label(pitch_deg: float, yaw_deg: float) -> str:
    """将俯仰角/偏航角（单位：度）转换为人类可读的注视方向字符串。"""
    v_thr, h_thr = 10.0, 10.0  # 垂直/水平方向判定阈值（度），超过则认为偏离正前方
    # 垂直方向：pitch > 0 向上，< 0 向下，在阈值内为 "Center"
    v = "Up" if pitch_deg > v_thr else ("Down" if pitch_deg < -v_thr else "Center")
    # 水平方向：yaw > 0 向右，< 0 向左，在阈值内为 "Center"
    h = "Right" if yaw_deg > h_thr else ("Left" if yaw_deg < -h_thr else "Center")
    if v == "Center" and h == "Center":  # 两个方向都在阈值内 -> 正前方
        return "Forward"
    if v == "Center":                    # 仅水平偏离 -> 返回左/右
        return h
    if h == "Center":                    # 仅垂直偏离 -> 返回上/下
        return v
    return f"{v}-{h}"                   # 斜向（如左上、右下等）


def _draw_semi_bg(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  color=(0, 0, 0), alpha: float = 0.55):
    """在帧上绘制半透明填充矩形，用作文字背景以提升可读性。"""
    x1, y1 = max(x, 0), max(y, 0)                   # 裁剪到帧左上边界内
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])  # 裁剪到帧右下边界内
    if x2 <= x1 or y2 <= y1:                         # 矩形在帧外（完全不可见），直接返回
        return
    roi = frame[y1:y2, x1:x2]                        # 取目标区域（ROI）
    overlay = roi.copy()                              # 复制 ROI 用于混合
    overlay[:] = color                                # 将覆盖层填充为指定颜色
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)  # alpha 混合：overlay*α + roi*(1-α)


def draw_info_overlay(frame: np.ndarray, results, fps: float, frame_idx: int = 0):
    """将所有注视推理结果绘制到帧上（全局统计面板 + 每张人脸信息面板）。"""
    h_frame, w_frame = frame.shape[:2]  # 获取帧的高度和宽度

    # ── 左上角：全局统计信息面板 ────────────────────────────────────────────
    n_faces = len(results.pitch) if results is not None else 0  # 当前帧检测到的人脸数量
    top_lines = [
        f"FPS: {fps:.1f}",        # 当前帧实时帧率
        f"Faces: {n_faces}",      # 检测到的人脸数
        f"Frame: {frame_idx}",    # 当前帧编号
    ]
    line_h = 24      # 每行文字占用的像素高度
    panel_w = 160    # 统计面板宽度（像素）
    panel_h = line_h * len(top_lines) + 8  # 面板总高度（行数 × 行高 + 边距）
    _draw_semi_bg(frame, 4, 4, panel_w, panel_h)  # 绘制半透明背景
    for li, txt in enumerate(top_lines):           # 遍历每行文字
        cy = 4 + 6 + li * line_h + line_h - 4     # 计算文字基线的 y 坐标
        if li == 0:                                # 第一行（FPS）使用绿色大字
            color = (0, 255, 0)
            scale, thick = 0.7, 2
        else:                                      # 其余行使用浅灰色小字
            color = (220, 220, 220)
            scale, thick = 0.55, 1
        cv2.putText(frame, txt, (8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if results is None or n_faces == 0:  # 无检测结果时直接返回，不绘制人脸信息
        return frame

    # ── 每张人脸的信息面板 ────────────────────────────────────────────────
    for i in range(n_faces):                        # 遍历每张检测到的人脸
        pitch_rad = float(results.pitch[i])         # 俯仰角（弧度），正值表示向上
        yaw_rad   = float(results.yaw[i])           # 偏航角（弧度），正值表示向右
        pitch_deg = pitch_rad * 180.0 / np.pi       # 转换为度数
        yaw_deg   = yaw_rad   * 180.0 / np.pi       # 转换为度数
        score     = float(results.scores[i])        # 人脸检测置信度分数
        bbox      = results.bboxes[i]               # 人脸边界框 [x_min, y_min, x_max, y_max]

        # 将边界框坐标裁剪到帧范围内，防止越界
        x_min = max(int(bbox[0]), 0)
        y_min = max(int(bbox[1]), 0)
        x_max = min(int(bbox[2]), w_frame)
        y_max = min(int(bbox[3]), h_frame)
        bw    = x_max - x_min   # 人脸框宽度（像素）
        bh    = y_max - y_min   # 人脸框高度（像素）

        direction = _gaze_direction_label(pitch_deg, yaw_deg)  # 计算注视方向文字

        # ── 构造该人脸的信息行列表（文字内容 + 颜色） ──────────────────
        info_lines = [
            (f"Face #{i+1}",                    (0, 255, 255)),   # 人脸编号，青色
            (f"Score:  {score:.3f}",             (200, 200, 200)), # 检测置信度，浅灰
            (f"Pitch:  {pitch_deg:+.2f} deg",    _angle_color(pitch_deg)),  # 俯仰角，颜色随角度变化
            (f"Yaw:    {yaw_deg:+.2f} deg",      _angle_color(yaw_deg)),    # 偏航角，颜色随角度变化
            (f"Dir:    {direction}",             (255, 200, 50)),  # 注视方向，橙黄色
            (f"BBox:   {bw}x{bh} px",           (180, 180, 180)), # 人脸框尺寸，灰色
            (f"  @ ({x_min},{y_min})",           (140, 140, 140)), # 人脸框左上角坐标，暗灰
        ]

        il_h    = 20                                # 信息面板每行高度（像素）
        il_w    = 190                               # 信息面板宽度（像素）
        total_h = il_h * len(info_lines) + 8       # 信息面板总高度

        # 优先将面板放在人脸框下方；若空间不足则放在上方
        px = x_min                                  # 面板水平起始位置与人脸框左边对齐
        if y_max + total_h + 4 < h_frame:          # 下方有足够空间
            py = y_max + 2                          # 面板顶部紧贴人脸框底部
        else:
            py = max(y_min - total_h - 2, 0)       # 面板底部紧贴人脸框顶部（边界保护）

        _draw_semi_bg(frame, px, py, il_w, total_h, color=(20, 20, 20), alpha=0.65)  # 深色半透明背景

        for li, (txt, col) in enumerate(info_lines):  # 逐行绘制信息文字
            cy = py + 6 + li * il_h + il_h - 4        # 当前行文字基线 y 坐标
            cv2.putText(frame, txt, (px + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        # ── 绘制 RetinaFace 输出的 5 个关键点（面部特征点） ───────────────
        if results.landmarks is not None and results.landmarks.shape[0] > i:
            lms = results.landmarks[i]  # 第 i 张人脸的关键点，形状 (5, 2)：[左眼, 右眼, 鼻, 左嘴角, 右嘴角]
            lm_labels = ["LE", "RE", "N", "LM", "RM"]  # 关键点缩写标签
            lm_colors = [
                (255, 100, 100),   # 左眼：蓝色系
                (100, 100, 255),   # 右眼：红色系
                (100, 255, 100),   # 鼻尖：绿色
                (255, 255, 100),   # 左嘴角：黄色
                (255, 100, 255),   # 右嘴角：紫色
            ]
            for li_idx, (lx, ly) in enumerate(lms):   # 遍历每个关键点
                cx, cy = int(lx), int(ly)              # 关键点像素坐标（取整）
                if 0 <= cx < w_frame and 0 <= cy < h_frame:  # 确认坐标在帧范围内
                    cv2.circle(frame, (cx, cy), 4, lm_colors[li_idx], -1, cv2.LINE_AA)  # 绘制实心圆点
                    cv2.putText(frame, lm_labels[li_idx],
                                (cx + 5, cy - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                lm_colors[li_idx], 1, cv2.LINE_AA)  # 在旁边标注缩写

    return frame  # 返回已绘制标注的帧


def _angle_color(deg: float):
    """根据角度绝对值返回颜色：0° 附近绿色，±20° 黄色，超过 ±40° 红色（BGR 格式）。"""
    abs_deg = abs(deg)
    if abs_deg < 20:        # 小角度：注视基本正前方，绿色表示正常
        return (80, 255, 80)
    if abs_deg < 40:        # 中等角度：注意力轻微偏离，黄色表示警告
        return (0, 200, 255)
    return (0, 80, 255)     # 大角度：注意力明显偏离，红色表示危险


def _step_with_scale(pipeline, frame: np.ndarray,
                     det_scale: float = 1.0,
                     max_faces: int = 0):
    """
    以可选的预缩放方式运行注视估计管线。

    det_scale < 1.0：在 RetinaFace 检测前缩小帧，显著提升高分辨率视频的检测速度。
    边界框和关键点坐标会被缩放回原始分辨率，再送入 L2CS 注视模型。
    注视模型始终从原始分辨率帧中裁剪人脸区域，保证精度。

    max_faces > 0：仅保留得分最高的 N 个结果，抑制多尺度重复检测。
    """
    from l2cs.results import GazeResultContainer  # 导入结果容器类

    orig_h, orig_w = frame.shape[:2]  # 原始帧的高度和宽度

    if det_scale < 1.0:                                      # 需要缩放时
        det_w = max(32, int(orig_w * det_scale))             # 缩放后宽度（至少 32 像素）
        det_h = max(32, int(orig_h * det_scale))             # 缩放后高度（至少 32 像素）
        det_frame = cv2.resize(frame, (det_w, det_h))        # 将帧缩小用于检测
        scale_x = orig_w / det_w                             # 水平方向缩放比（坐标还原用）
        scale_y = orig_h / det_h                             # 垂直方向缩放比（坐标还原用）
    else:                                                    # 不缩放时直接使用原帧
        det_frame = frame
        scale_x = scale_y = 1.0                             # 坐标无需缩放

    # 初始化结果列表
    face_imgs  = []   # 裁剪并预处理后的人脸图像列表（送入注视模型）
    bboxes     = []   # 人脸边界框列表（已还原到原始分辨率）
    landmarks  = []   # 关键点列表（已还原到原始分辨率）
    scores     = []   # 检测置信度分数列表

    faces = pipeline.detector(det_frame)  # 在（可能缩小的）帧上运行人脸检测器

    if faces is not None:                 # 有检测结果时处理每张人脸
        for box, landmark, score in faces:
            if score < pipeline.confidence_threshold:  # 过滤低置信度检测结果
                continue

            # 将边界框坐标从检测帧还原到原始帧尺寸
            x_min = max(int(box[0] * scale_x), 0)
            y_min = max(int(box[1] * scale_y), 0)
            x_max = min(int(box[2] * scale_x), orig_w)
            y_max = min(int(box[3] * scale_y), orig_h)

            if x_max <= x_min or y_max <= y_min:  # 跳过无效（零面积）边界框
                continue

            # 从原始帧裁剪人脸区域，保证注视模型输入分辨率足够高
            crop = frame[y_min:y_max, x_min:x_max]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # BGR 转 RGB（模型期望 RGB 输入）
            crop = cv2.resize(crop, (224, 224))            # 缩放到模型输入尺寸 224x224
            face_imgs.append(crop)

            # 保存已还原的边界框（float32 数组）
            scaled_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            bboxes.append(scaled_box)

            # 将关键点坐标从检测帧还原到原始帧尺寸
            scaled_lm = landmark.copy().astype(np.float32)
            scaled_lm[:, 0] *= scale_x   # x 坐标还原
            scaled_lm[:, 1] *= scale_y   # y 坐标还原
            landmarks.append(scaled_lm)
            scores.append(score)

    # 若设置了最大人脸数，按置信度降序只保留前 N 张（抑制重复检测）
    if max_faces > 0 and len(scores) > max_faces:
        order = np.argsort(scores)[::-1][:max_faces]       # 取得分最高的 N 个索引
        face_imgs  = [face_imgs[i]  for i in order]        # 按索引筛选人脸图像
        bboxes     = [bboxes[i]     for i in order]        # 按索引筛选边界框
        landmarks  = [landmarks[i]  for i in order]        # 按索引筛选关键点
        scores     = [scores[i]     for i in order]        # 按索引筛选置信度

    if len(face_imgs) > 0:                                 # 有有效人脸时运行注视估计
        pitch, yaw = pipeline.predict_gaze(np.stack(face_imgs))  # 批量推理，返回俯仰角和偏航角
    else:                                                  # 无人脸时返回空结果
        pitch = np.empty((0,))  # 空俯仰角数组
        yaw   = np.empty((0,))  # 空偏航角数组

    # 封装并返回结果容器（无结果时用空数组占位，保证形状正确）
    return GazeResultContainer(
        pitch=pitch,
        yaw=yaw,
        bboxes=np.stack(bboxes)       if len(bboxes)     > 0 else np.empty((0, 4)),    # 边界框 (N,4)
        landmarks=np.stack(landmarks) if len(landmarks)  > 0 else np.empty((0, 5, 2)), # 关键点 (N,5,2)
        scores=np.array(scores)       if len(scores)     > 0 else np.empty((0,)),       # 置信度 (N,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()  # 解析命令行参数

    # ── 确定模型权重文件路径 ──────────────────────────────────────────────
    if args.weights:                                         # 用户指定了权重路径
        weights_path = pathlib.Path(args.weights)
    else:                                                    # 使用默认权重路径
        weights_path = ROOT / "models" / "L2CSNet_gaze360.pkl"

    if not weights_path.exists():                            # 权重文件不存在时报错退出
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Run:  python download_model.py")      # 提示用户先下载模型
        sys.exit(1)

    # ── 解析推理设备字符串 ────────────────────────────────────────────────
    device_str = args.device.lower()
    # 将 'gpu:0' -> '0'，便于 select_device 识别
    if device_str.startswith("gpu:"):
        device_str = device_str[4:]   # 截取冒号后的数字部分
    elif device_str == "gpu":
        device_str = "0"              # 未指定卡号时默认使用 GPU 0
    # 'cpu' 保持不变，直接传入 select_device

    cudnn.enabled = True              # 启用 cuDNN 加速（GPU 模式下生效）
    device = select_device(device_str, batch_size=1)  # 创建 PyTorch 设备对象
    print(f"[INFO] Device: {device}")

    # ── 初始化注视估计管线 ────────────────────────────────────────────────
    det_scale  = max(0.1, min(1.0, args.det_scale))   # 将检测缩放比约束在 [0.1, 1.0] 内
    max_faces  = args.max_faces                        # 每帧保留的最大人脸数
    print(f"[INFO] Loading model: {weights_path}")
    print(f"[INFO] Confidence threshold : {args.confidence}")
    print(f"[INFO] Max faces per frame  : {max_faces if max_faces > 0 else 'unlimited'}")
    print(f"[INFO] Detection scale      : {det_scale:.2f}  (detector input = original x{det_scale:.2f})")
    gaze_pipeline = Pipeline(
        weights=weights_path,                # 模型权重文件路径
        arch=args.arch,                      # 骨干网络结构
        device=device,                       # 推理设备
        confidence_threshold=args.confidence,# 人脸检测置信度阈值
    )
    print("[INFO] Model loaded.")  # 模型加载完成提示

    # ── 打开输入视频源 ────────────────────────────────────────────────────
    cap, is_cam = open_source(args.input)   # 打开摄像头或视频文件
    if not cap.isOpened():                  # 打开失败时报错退出
        print(f"[ERROR] Cannot open input: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0  # 视频总帧数（摄像头为 0）
    source_label = "webcam" if is_cam else args.input                            # 输入源描述文字
    print(f"[INFO] Input: {source_label}")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")  # 仅对视频文件打印总帧数

    # ── 创建输出视频写入器 ────────────────────────────────────────────────
    writer = None
    if not args.no_save:          # 未指定 --no-save 时才创建写入器
        writer = make_writer(args.output, cap)

    # ── 尝试创建显示窗口 ──────────────────────────────────────────────────
    show = args.show              # 是否实时显示结果窗口
    if show:
        # 尝试创建窗口以检测是否有可用显示环境
        try:
            cv2.namedWindow("L2CS-Net Gaze", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[WARN] No display environment detected, disabling --show")  # 无显示环境时禁用
            show = False

    # ── 主处理循环 ────────────────────────────────────────────────────────
    frame_idx   = 0      # 已处理帧计数器
    fps_display = 0.0    # 当前显示帧率（实时更新）
    t_start     = time.time()  # 整体处理开始时间（用于计算平均帧率）

    print("[INFO] Processing... Press 'q' to quit (if --show is enabled).")

    with torch.no_grad():   # 关闭梯度计算，节省显存和内存，提升推理速度
        while True:
            ret, frame = cap.read()   # 从视频源读取一帧
            if not ret:               # 读取失败
                if is_cam:
                    print("[WARN] Failed to read from camera, retrying...")  # 摄像头短暂断帧时重试
                    time.sleep(0.05)
                    continue
                else:
                    print("[INFO] End of video.")  # 视频文件读完，正常退出循环
                    break

            frame_idx += 1  # 帧计数器自增
            if args.max_frames > 0 and frame_idx > args.max_frames:  # 达到最大帧数限制
                print(f"[INFO] Reached max-frames limit ({args.max_frames}).")
                break

            t_frame = time.time()  # 当前帧处理开始时间（用于计算单帧 FPS）

            # ── 推理：人脸检测 + 注视估计 ────────────────────────────────
            try:
                results = _step_with_scale(gaze_pipeline, frame, det_scale, max_faces)
            except Exception as e:
                print(f"[WARN] Frame {frame_idx}: inference error: {e}")  # 推理出错时跳过此帧
                results = None

            fps_display = 1.0 / max(time.time() - t_frame, 1e-6)  # 计算当前帧率（避免除零）

            # ── 可视化：在帧上绘制标注 ───────────────────────────────────
            if results is not None and len(results.pitch) > 0:
                frame = render(frame, results)                              # 绘制注视箭头和人脸框
                frame = draw_info_overlay(frame, results, fps_display, frame_idx)  # 绘制信息面板
            else:
                # 未检测到人脸时仅绘制全局统计面板
                frame = draw_info_overlay(frame, results, fps_display, frame_idx)

            # ── 写入输出视频 ──────────────────────────────────────────────
            if writer is not None:
                writer.write(frame)   # 将标注后的帧写入输出视频文件

            # ── 实时显示 ──────────────────────────────────────────────────
            if show:
                cv2.imshow("L2CS-Net Gaze", frame)  # 显示当前帧
                key = cv2.waitKey(1) & 0xFF          # 等待 1ms 按键事件
                if key == ord("q"):                  # 按下 'q' 键时退出
                    print("[INFO] 'q' pressed, stopping.")
                    break

            # ── 每 30 帧打印一次进度日志 ──────────────────────────────────
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start              # 累计耗时
                faces = len(results.pitch) if results is not None else 0  # 当前帧人脸数
                if total_frames > 0:                         # 视频文件：显示进度百分比
                    pct = frame_idx / total_frames * 100
                    print(
                        f"[INFO] Frame {frame_idx}/{total_frames} ({pct:.1f}%)  "
                        f"FPS: {fps_display:.1f}  Faces: {faces}"
                    )
                else:                                        # 摄像头：仅显示帧号
                    print(
                        f"[INFO] Frame {frame_idx}  "
                        f"FPS: {fps_display:.1f}  Faces: {faces}"
                    )

    # ── 释放资源 ──────────────────────────────────────────────────────────
    cap.release()            # 释放视频捕获对象
    if writer is not None:
        writer.release()     # 刷新缓冲区并关闭输出视频文件
    if show:
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 显示窗口

    # 打印整体处理统计信息
    total_time = time.time() - t_start
    avg_fps    = frame_idx / max(total_time, 1e-6)  # 平均帧率（避免除零）
    print(f"\n[DONE] Processed {frame_idx} frames in {total_time:.1f}s (avg {avg_fps:.1f} FPS)")
    if writer is not None:
        out_path = pathlib.Path(args.output)
        size_mb  = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0  # 输出文件大小（MB）
        print(f"[DONE] Output saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()  # 脚本直接运行时调用主函数
