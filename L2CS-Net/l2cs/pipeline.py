import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer

# ── 兼容两种 face-detection 版本 ───────────────────────────────────────────
# v0.2.x (elliottzheng)：通过 build_detector 创建，支持多模型选择，需联网下载权重
# v1.0.x (第三方同名包)：RetinaFace 类直接可用，MobileNet0.25 权重打包在包内，无需联网
try:
    from face_detection import build_detector as _build_detector
    _FACE_DET_BACKEND = "elliottzheng"  # 0.2.x：支持 RetinaNetResNet50 / MobileNetV1 / DSFD
except ImportError:
    _build_detector = None
    _FACE_DET_BACKEND = None

try:
    from face_detection import RetinaFace as _RetinaFaceLegacy
    if _FACE_DET_BACKEND is None:
        _FACE_DET_BACKEND = "legacy"      # 1.0.x：只有 MobileNet0.25，权重打包在包内
except ImportError:
    _RetinaFaceLegacy = None

if _FACE_DET_BACKEND is None:
    raise ImportError(
        "No supported face_detection package found. "
        "Install via:  pip install face-detection==0.2.2"
    )


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5,
        face_detector: str = 'RetinaNetResNet50'
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.face_detector = face_detector
        self._backend = _FACE_DET_BACKEND

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create face detector if requested
        if self.include_detector:
            if self._backend == "elliottzheng":
                # 0.2.x：通过 build_detector 创建，支持三种模型
                self.detector = _build_detector(
                    name=face_detector,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
            else:
                # 1.0.x：直接实例化，只有 MobileNet0.25，gpu_id=-1 表示 CPU
                gpu_id = -1
                if hasattr(device, 'index') and device.index is not None:
                    gpu_id = device.index
                elif isinstance(device, str) and device.startswith('cuda'):
                    parts = device.split(':')
                    gpu_id = int(parts[1]) if len(parts) > 1 else 0
                self.detector = _RetinaFaceLegacy(gpu_id=gpu_id)
                if face_detector != 'RetinaNetResNet50':
                    print(f"[WARN] face-detection legacy backend only supports MobileNet0.25, "
                          f"ignoring --face-detector={face_detector}")

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def detect_faces(self, frame: np.ndarray):
        """Run face detector on a BGR uint8 frame.

        Returns a list of (box [4], landmark [5,2], score) tuples, or None.

        Backend: elliottzheng 0.2.x
          - RetinaNetResNet50    : ~27M params, landmarks, 需联网下载权重
          - RetinaNetMobileNetV1 : ~0.4M params, landmarks, 需联网下载权重
          - DSFDDetector         : ~120M params, 无关键点, 需联网下载权重
        Backend: legacy 1.0.x
          - MobileNet0.25, 权重打包在包内, 无需联网
        """
        if self._backend == "elliottzheng":
            return self._detect_elliottzheng(frame)
        else:
            return self._detect_legacy(frame)

    def _detect_elliottzheng(self, frame: np.ndarray):
        """使用 elliottzheng 0.2.x 后端检测（batched_detect_with_landmarks / detect）。"""
        has_landmarks = self.face_detector != 'DSFDDetector'
        if has_landmarks:
            boxes_list, landmarks_list = self.detector.batched_detect_with_landmarks(
                frame[None]  # [1, H, W, 3]
            )
            dets = boxes_list[0]      # [N, 5]: x1 y1 x2 y2 score
            lms  = landmarks_list[0]  # [N, 5, 2]
            if len(dets) == 0:
                return None
            return [(dets[j, :4], lms[j], float(dets[j, 4])) for j in range(len(dets))]
        else:
            dets = self.detector.detect(frame)  # [N, 5]
            if len(dets) == 0:
                return None
            empty_lm = np.zeros((5, 2), dtype=np.float32)
            return [(dets[j, :4], empty_lm.copy(), float(dets[j, 4])) for j in range(len(dets))]

    def _detect_legacy(self, frame: np.ndarray):
        """使用 legacy 1.0.x 后端检测（RetinaFace.__call__ 返回列表）。"""
        faces = self.detector(frame)  # list of (box[4], landmark[5,2], score)
        if faces is None or len(faces) == 0:
            return None
        result = []
        for box, landmark, score in faces:
            if float(score) < self.confidence_threshold:
                continue
            result.append((np.asarray(box, dtype=np.float32),
                           np.asarray(landmark, dtype=np.float32),
                           float(score)))
        return result if result else None

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detect_faces(frame)

            if faces is not None:
                for box, landmark, score in faces:

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze (only when at least one face passed threshold)
                if len(face_imgs) > 0:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                else:
                    pitch = np.empty((0,))
                    yaw   = np.empty((0,))

            else:
                pitch = np.empty((0,))
                yaw   = np.empty((0,))

        else:
            pitch, yaw = self.predict_gaze(frame)

        # Save data — use empty arrays with correct shape when nothing detected
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes)     if len(bboxes)     > 0 else np.empty((0, 4)),
            landmarks=np.stack(landmarks) if len(landmarks) > 0 else np.empty((0, 5, 2)),
            scores=np.stack(scores)     if len(scores)     > 0 else np.empty((0,)),
        )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted
