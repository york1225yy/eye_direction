# L2CS-Net 视线估计 — 完整使用说明

基于 [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) 的驾驶员视线方向估计，支持视频文件输入与摄像头实时推理，输出带注视箭头标注的 MP4 视频。

---

## 目录

1. [项目结构](#项目结构)
2. [环境要求](#环境要求)
3. [一键环境配置（AutoDL）](#一键环境配置autodl)
4. [手动安装](#手动安装)
5. [下载预训练模型](#下载预训练模型)
6. [运行 Demo](#运行-demo)
7. [参数说明](#参数说明)
8. [输出说明](#输出说明)
9. [常见问题](#常见问题)

---

## 项目结构

```
L2CS-Net/
├── l2cs/                   # 模型核心包
│   ├── model.py            # L2CS 网络定义（ResNet backbone）
│   ├── pipeline.py         # 推理管线（RetinaFace 检测 + 注视估计）
│   ├── vis.py              # 可视化工具（绘制箭头、边框）
│   ├── utils.py            # 工具函数（设备选择、预处理等）
│   └── results.py          # 结果数据结构
├── models/                 # 预训练模型存放目录
│   └── L2CSNet_gaze360.pkl # ← 运行前需下载
├── datasets/
│   └── result_long.mp4     # 测试视频
├── output/                 # 结果视频输出目录（自动创建）
├── run_demo.py             # 本项目主 Demo 脚本（功能增强版）
├── download_model.py       # 模型自动下载脚本
├── setup_env.sh            # 一键环境配置脚本（AutoDL）
├── demo.py                 # 原始官方 Demo（仅摄像头）
└── pyproject.toml          # 包依赖声明
```

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.8 | 推荐 3.9 / 3.10 |
| PyTorch | ≥ 1.10 | CPU 或 CUDA 版均可 |
| torchvision | ≥ 0.11 | 与 PyTorch 版本对应 |
| opencv-python | ≥ 4.5 | 视频读写 |
| face_detection | 最新 | RetinaFace 人脸检测 |
| numpy / scipy / Pillow / pandas / matplotlib | 见 pyproject.toml | — |

---

## 一键环境配置（AutoDL）

> 在 AutoDL 实例的终端中执行以下命令，脚本会**自动检测已安装的包并跳过**，避免重复安装。

```bash
# 进入项目目录
cd /root/L2CS-Net          # 根据实际路径修改

# 赋予执行权限并运行
chmod +x setup_env.sh
bash setup_env.sh
```

脚本执行流程：

```
1. 检测 Python 版本（≥ 3.8）
2. 升级 pip / setuptools / wheel
3. 检测 GPU 和 CUDA 版本 → 自动选择对应 PyTorch wheel 安装
4. 逐个检测并安装 L2CS-Net 依赖（已安装则跳过）
5. 以 editable 模式安装 l2cs 包
6. 自动下载预训练模型 L2CSNet_gaze360.pkl
```

完成后脚本会打印验证信息和快速测试命令。

---

## 手动安装

### Step 1：安装 PyTorch

**CPU 版：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8 版（AutoDL 常用）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1 版：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2：安装其余依赖

```bash
pip install matplotlib numpy opencv-python pandas Pillow scipy gdown
pip install git+https://github.com/elliottzheng/face-detection
```

### Step 3：安装 l2cs 包

```bash
cd L2CS-Net
pip install -e .
```

---

## 下载预训练模型

模型文件 `L2CSNet_gaze360.pkl`（~216 MB）使用 Gaze360 数据集训练，Angular Error ≈ 4.0°。

### 方式一：自动下载（推荐，支持国内环境）

脚本按优先级依次尝试以下来源，**国内/AutoDL 环境会自动走 hf-mirror.com 镜像**：

| 优先级 | 来源 | 可用环境 |
|--------|------|---------|
| 1 | HuggingFace 镜像 (hf-mirror.com) | AutoDL / 国内 ✅ |
| 2 | HuggingFace 官方 (huggingface.co) | 有梯子 / 海外 ✅ |
| 3 | Google Drive (gdown) | 有梯子 / 海外 ✅ |

```bash
python download_model.py
```

强制重新下载：
```bash
python download_model.py --force
```

### 方式二：手动上传（AutoDL 无法自动下载时）

**Option A — AutoDL 控制台文件上传：**
1. 在本地浏览器打开 Google Drive 或 HuggingFace 下载模型：  
   https://huggingface.co/oraclex/L2CS-Net  
   或 https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd
2. 登录 AutoDL 控制台 → 实例详情 → **文件上传**
3. 上传 `L2CSNet_gaze360.pkl` 后在终端执行：
   ```bash
   mv ~/L2CSNet_gaze360.pkl /path/to/L2CS-Net/models/
   ```

**Option B — JupyterLab 上传：**
1. 打开 AutoDL JupyterLab
2. 切换到 `L2CS-Net/models/` 目录
3. 点击工具栏上传按钮（↑）上传文件

**Option C — scp 从本地传入：**
```bash
scp L2CSNet_gaze360.pkl root@<autodl-host>:<ssh-port>:/path/to/L2CS-Net/models/
```

---

## 运行 Demo

所有命令在 `L2CS-Net/` 目录下执行。

### 视频文件 + CPU

```bash
python run_demo.py \
  --input  datasets/result_long.mp4 \
  --device cpu \
  --output output/result_cpu.mp4
```

### 视频文件 + GPU 0

```bash
python run_demo.py \
  --input  datasets/result_long.mp4 \
  --device gpu:0 \
  --output output/result_gpu.mp4
```

### 摄像头 + CPU

```bash
python run_demo.py \
  --input  cam \
  --device cpu \
  --output output/webcam.mp4
```

### 摄像头 + GPU，同时显示窗口

```bash
python run_demo.py \
  --input  cam \
  --device gpu:0 \
  --output output/webcam.mp4 \
  --show
```

### 只保存前 300 帧（快速验证）

```bash
python run_demo.py \
  --input      datasets/result_long.mp4 \
  --device     cpu \
  --output     output/sample300.mp4 \
  --max-frames 300
```

### 不保存视频（仅屏幕显示）

```bash
python run_demo.py \
  --input   cam \
  --device  cpu \
  --no-save \
  --show
```

---

## 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `cam` | 输入源：`cam` / `cam:N` / 视频文件路径 |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `gpu:0` / `gpu:1` 等 |
| `--arch` | — | `ResNet50` | 骨干网络：ResNet18/34/50/101/152 |
| `--weights` | `-w` | `models/L2CSNet_gaze360.pkl` | 模型权重路径 |
| `--output` | `-o` | `output/gaze_result.mp4` | 输出视频路径 |
| `--confidence` | `-c` | `0.5` | 人脸检测置信度阈值（0~1） |
| `--show` | — | off | 实时显示窗口（需要图形环境） |
| `--no-save` | — | off | 不保存输出视频 |
| `--max-frames` | — | `0`（无限制） | 最多处理的帧数 |

---

## 输出说明

### 视频标注内容

| 元素 | 颜色 | 含义 |
|------|------|------|
| 边框矩形 | 绿色 | 检测到的人脸区域 |
| 注视箭头 | 红色 | 视线方向向量（Pitch + Yaw） |
| `P: / Y:` 文字 | 白色 | 俯仰角（Pitch）/ 偏航角（Yaw），单位：度 |
| `FPS` 计数 | 绿色 | 当前帧推理帧率 |

### 角度含义

```
Pitch（俯仰角）：
  正值 → 抬头（向上看）
  负值 → 低头（向下看）

Yaw（偏航角）：
  正值 → 向右看
  负值 → 向左看
```

### 输出文件

处理完成后结果保存为 `.mp4` 文件，使用 `mp4v` 编码，可直接用 VLC / 浏览器播放。

---

## 常见问题

### Q1：`ModuleNotFoundError: No module named 'l2cs'`

```bash
# 进入 L2CS-Net 目录重新安装
cd L2CS-Net
pip install -e .
```

### Q2：`ModuleNotFoundError: No module named 'face_detection'`

```bash
pip install git+https://github.com/elliottzheng/face-detection
```

### Q3：模型下载失败（Google Drive 限流）

手动从此链接下载并放到 `models/` 目录：  
https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd

```bash
# 验证文件大小应 > 200MB
ls -lh models/L2CSNet_gaze360.pkl
```

### Q4：`Cannot open webcam` / 摄像头无法打开

- 服务器/AutoDL 实例通常无摄像头，请改用视频文件：  
  `--input datasets/result_long.mp4`
- 如需测试摄像头，确认摄像头设备存在：`ls /dev/video*`

### Q5：GPU 模式报错 `CUDA unavailable`

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
若输出 `False`，说明 PyTorch 安装的是 CPU 版，需重装对应 CUDA 版本。

### Q6：输出视频无法播放

```bash
# 用 ffmpeg 重新编码为 H.264（兼容性更好）
ffmpeg -i output/result_cpu.mp4 -vcodec libx264 output/result_h264.mp4
```

### Q7：推理速度慢（CPU 下 < 5 FPS）

可换更轻量的骨干网络：

```bash
python run_demo.py --input datasets/result_long.mp4 --arch ResNet18 --device cpu --output output/result_r18.mp4
```

---

## 参考

- 论文：[L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://arxiv.org/abs/2203.03339)
- 原始仓库：https://github.com/Ahmednull/L2CS-Net
- Gaze360 数据集：http://gaze360.csail.mit.edu/
