# 车载驾驶员监控系统视线估计算法调研报告

> 调研日期：2026-05-08

---

## 目录

1. [项目概览对比表](#一项目概览对比表)
2. [GazeTracking](#二gazetracking)
3. [L2CS-Net](#三l2cs-net)
4. [RT-GENE](#四rt-gene)
5. [ETH-XGaze](#五eth-xgaze)
6. [pytorch_mpiigaze（MPIIGaze/MPIIFaceGaze PyTorch 实现）](#六pytorch_mpiigaze)
7. [GazeML](#七gazeml)
8. [综合评估与驾驶员监控推荐](#八综合评估与驾驶员监控推荐)

---

## 一、项目概览对比表

| 项目 | GitHub | 框架 | 方法类型 | 训练数据集 | 典型角度误差 | License | Stars | 最近更新 | 驾驶监控适用性 |
|------|--------|------|---------|-----------|-------------|---------|-------|---------|--------------|
| GazeTracking | [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking) | OpenCV + dlib | 瞳孔检测（传统CV） | 无（dlib预训练） | 无量化指标（仅左/中/右分类） | MIT | ⭐2,564 | 2026-05 | ⚠️ 低（无角度输出，头姿无关） |
| L2CS-Net | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | PyTorch | 深度学习（ResNet+分类+回归） | Gaze360, MPIIFaceGaze | ~4.0°（Gaze360）/ ~4.2°（MPII） | MIT-like | ⭐498 | 2026-05 | ✅ 高（pitch/yaw，实时，有预训练模型） |
| RT-GENE | [Tobias-Fischer/rt_gene](https://github.com/Tobias-Fischer/rt_gene) | TF / PyTorch | 深度学习（VGG/ResNet双眼+头姿） | RT-GENE, MPIIGaze, UTMV | ~6.3°（MPII集成） | CC BY-NC-SA 4.0 | ⭐439 | 2026-05 | ⚠️ 中（非商业，需ROS依赖） |
| ETH-XGaze | [xucong-zhang/ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) | PyTorch | 深度学习（ResNet50+数据归一化） | ETH-XGaze（>100万样本） | ~5.0°（ETH-XGaze测试集） | CC BY-NC-SA 4.0 | ⭐227 | 2026-03 | ✅ 高（极端头姿鲁棒，适合驾驶场景） |
| pytorch_mpiigaze | [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze) | PyTorch | 深度学习（LeNet/VGG，外观回归） | MPIIGaze, MPIIFaceGaze | ~4.5–5.0°（MPII LOO） | MIT | ⭐379 | 2026-05 | ⚠️ 中（研究基线，不含完整推理管线） |
| GazeML | [swook/GazeML](https://github.com/swook/GazeML) | TensorFlow 1.x | 深度学习（ELG眼部关键点+凝视） | MPIIGaze | ~5.6°（MPIIGaze） | MIT | ⭐579 | 2026-04 | ❌ 低（TF1，无完整人脸流水线） |

---

## 二、GazeTracking

### 基本信息

- **GitHub URL**：https://github.com/antoinelame/GazeTracking
- **Stars / Forks**：⭐ 2,564 / 🍴 623（截至 2026-05-08，仍在积极维护）
- **License**：MIT（可商用）
- **语言**：Python 3.10+
- **依赖**：OpenCV、dlib、NumPy（无 PyTorch/TensorFlow）

### 方法与原理

纯传统计算机视觉方案，**无需深度学习**：

1. **人脸检测**：dlib HOG 正面人脸检测器 (`get_frontal_face_detector`)
2. **人脸关键点**：dlib 68 点关键点预测模型 (`shape_predictor_68_face_landmarks.dat`)，利用 36–41（左眼）、42–47（右眼）点位提取眼部区域
3. **虹膜/瞳孔检测**：
   - 对眼部区域进行自适应阈值二值化（`Calibration` 类在前20帧内自动调参）
   - 双边滤波 + 腐蚀 + 轮廓检测 → 矩心法定位瞳孔坐标 `(x, y)`
4. **视线方向判断**：
   - `horizontal_ratio = pupil_x / (eye_center_x * 2)`，返回 `[0.0, 1.0]`（0=右，0.5=中，1=左）
   - `vertical_ratio` 同理
   - 阈值判断：`≤ 0.35` → looking right；`≥ 0.65` → looking left
5. **眨眼检测**：眼宽 / 眼高比率 `> 3.8` 判为眨眼

### 性能指标

| 指标 | 值 |
|------|-----|
| 角度误差（度） | **无量化**（仅定性：左/中/右/眨眼） |
| 推理速度 | 实时（CPU 可运行） |
| 头姿鲁棒性 | 差（对头部旋转无补偿） |

### 优点

- ✅ 极低依赖，纯 CPU 运行，无需 GPU
- ✅ MIT 协议，可商用
- ✅ 代码极简清晰，易于集成和二次开发
- ✅ 自动自校准（无需用户标定）
- ✅ 内置眨眼检测

### 缺点

- ❌ 无角度量化输出（不能给出 pitch/yaw 度数）
- ❌ 头姿无关：头部旋转会严重影响精度
- ❌ 强光/弱光、眼镜对检测影响大
- ❌ 仅分类（左/中/右），不适合精细注视点追踪
- ❌ 对非正面或侧脸无法工作

### 驾驶员监控适用性

**⚠️ 低**。适合快速原型验证，不建议用于量产驾驶员监控。原因：
- 无法输出视线角度（无法判断驾驶员视线偏离量）
- 头部姿态不被考虑（驾驶员低头、侧头时失效）
- 不支持眼镜

---

## 三、L2CS-Net

### 基本信息

- **GitHub URL**：https://github.com/Ahmednull/L2CS-Net
- **Stars / Forks**：⭐ 498 / 🍴 108（截至 2026-05-08）
- **License**：MIT-like（README 中未明确声明，代码结构开放）
- **Paper**：_L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments_ (2022)
- **依赖**：Python, PyTorch, torchvision, OpenCV, RetinaFace

### 方法与原理

**基于 ResNet 的深度分类+回归混合方案**：

1. **人脸检测**：RetinaFace 检测器（检测人脸 bounding box 和关键点）
2. **人脸裁剪**：将检测到的人脸区域 resize 到 448×448，ImageNet 归一化
3. **骨干网络**：ResNet-50（默认），支持 ResNet-18/34/101/152
4. **输出头**：两个独立的全连接层，分别预测 pitch（俯仰）和 yaw（偏航）
5. **损失函数（L2CS：Look-to-Classify-and-Softmax）**：
   - **分类损失**：将 ±90° 范围均匀划分为 90 个 bin，使用交叉熵 `CrossEntropyLoss`
   - **回归损失**：对 Softmax 加权求和后与连续标签做 MSELoss
   - 总损失：`L_total = L_ce_pitch + α * L_mse_pitch + L_ce_yaw + α * L_mse_yaw`（α=1）
6. **推理**：对 Softmax 概率分布加权求和 → 连续 pitch/yaw 角度（弧度）

```python
# 推理示例
from l2cs import Pipeline, render
import torch, cv2, pathlib

pipeline = Pipeline(
    weights=pathlib.Path('models/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device('cpu')
)
cap = cv2.VideoCapture(0)
_, frame = cap.read()
results = pipeline.step(frame)  # returns GazeResultContainer(pitch, yaw, bboxes, ...)
```

### 性能指标

| 数据集 | 评估协议 | 平均角度误差 |
|--------|---------|------------|
| Gaze360 | train-val-test split | ~4.00° |
| MPIIFaceGaze | leave-one-person-out | ~4.24° |

（数据来自原论文，ResNet-50 backbone）

### 优点

- ✅ **完整推理 Pipeline**：内置 RetinaFace 人脸检测→裁剪→视线预测，一行代码可运行
- ✅ 输出连续 pitch/yaw 角度（弧度），适合量化分析
- ✅ 提供预训练模型（Gaze360 训练 ResNet-50）
- ✅ 支持多人同帧检测
- ✅ 可选多种 ResNet 骨干，在精度和速度间灵活权衡
- ✅ 活跃维护，已打包为 pip 可安装库

### 缺点

- ❌ **无头姿补偿**：模型输入为裁剪人脸图像，未显式建模头部姿态
- ❌ 需要 GPU 以实现最佳实时推理速度（CPU 速度较慢）
- ❌ 对极端头姿（± 80°+）精度下降（Gaze360 仅过滤 ±180° 范围）
- ❌ 预训练模型为 `pkl` 格式，需确保 PyTorch 版本兼容

### 驾驶员监控适用性

**✅ 高**。优先推荐用于驾驶员注视行为研究和监控原型开发：
- 输出精确 pitch/yaw（可判断视线偏离方向和角度）
- 一键式推理管线适合快速部署
- 可实时处理（GPU 环境下）
- 支持多乘员（可检测多人）

---

## 四、RT-GENE

### 基本信息

- **GitHub URL**：https://github.com/Tobias-Fischer/rt_gene
- **Stars / Forks**：⭐ 439 / 🍴 74（截至 2026-05-08）
- **License**：CC BY-NC-SA 4.0（**非商业，不可用于量产**）
- **Paper**：_RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments_（ECCV 2018）
- **依赖**：TensorFlow / PyTorch（二选一），ROS（可选，提供 standalone 版本），OpenCV, dlib

### 方法与原理

**双眼图像 + 头姿信息的多输入 CNN 集成方案**：

1. **人脸及关键点检测**：3DDFA（三维变形人脸模型）提取 68 关键点 + 头部姿态
2. **眼部区域提取**：根据关键点裁剪左眼/右眼图像（60×36 像素）
3. **眼部修复（Inpainting）**：使用 GAN 擦除眼镜遮挡，增强鲁棒性（可选模块）
4. **视线估计网络**：
   - **双路 CNN**：左眼、右眼分别通过 ResNet-18/VGG-16/PreActResNet 提取特征（1024-d）
   - **融合层**：Concat(左特征, 右特征) → 512-d → 合并头姿 2-d → 256-d → 输出 2-d（pitch, yaw）
5. **集成推理**：使用 4 个模型的平均值（训练于 RT-GENE + MPII + UTMV 数据集），以提升精度

### 性能指标

| 数据集 | 方法 | 平均角度误差 |
|--------|------|------------|
| MPIIGaze | 单模型 ResNet18 | ~8.8° |
| MPIIGaze | 4-模型集成（papers-with-code 排名） | ~6.3° |
| RT-GENE 自有 | 多种配置 | 参见论文 Table |
| UTMV | 集成 | 上榜 SOTA |

### 优点

- ✅ **眼镜友好**：独有的眼部修复（inpainting）模块，显著提升戴眼镜场景鲁棒性（驾驶员常戴眼镜！）
- ✅ 双眼融合 + 头姿信息，生理上更合理
- ✅ 支持 PyTorch 和 TensorFlow 双后端
- ✅ 配套完整数据集（RT-GENE Dataset 在 Zenodo 开放）
- ✅ 同时提供眨眼检测（RT-BENE）

### 缺点

- ❌ **CC BY-NC-SA 4.0，不可商用**（车厂量产须获授权）
- ❌ 全功能版本依赖 ROS 框架（嵌入式部署困难）
- ❌ standalone 版功能受限，多模型集成配置复杂
- ❌ 精度在不集成的情况下（单模型）较 L2CS-Net 差
- ❌ 需要多输入（双眼 + 头姿），推理链路较长

### 驾驶员监控适用性

**⚠️ 中**。研究价值很高，尤其在眼镜场景下。但非商业协议和 ROS 依赖限制了量产部署。适合学术研究和算法探索。

---

## 五、ETH-XGaze

### 基本信息

- **GitHub URL**：https://github.com/xucong-zhang/ETH-XGaze
- **Stars / Forks**：⭐ 227 / 🍴 ~40（截至 2026-03-29）
- **License**：CC BY-NC-SA 4.0（**非商业**）
- **Paper**：_ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation_（ECCV 2020）
- **依赖**：Python 3.5+, PyTorch 1.1+, OpenCV, dlib, h5py

### 方法与原理

**ResNet-50 + 数据归一化（头姿感知的人脸裁剪）**：

1. **人脸检测 + 关键点**：dlib 检测人脸，68 点关键点
2. **头部姿态估计**：PnP 算法（`solvePnP`）从 6 个关键点估计头部旋转向量 `rvec` 和平移 `tvec`
3. **数据归一化**（核心贡献）：
   - 构造旋转矩阵使相机轴与人脸轴对齐，消除头部转动影响
   - 归一化裁剪人脸图像（448×448），焦距归一化到 960px，距离归一化到 300mm
   - 同步归一化 gaze vector
4. **骨干网络**：ResNet-50（预训练 ImageNet）
   - 全连接输出层：`Linear(2048, 2)` → pitch, yaw
5. **损失函数**：L1 Loss（`angular_error` 监控训练效果）
6. **数据集**：ETH-XGaze（110 名受试者，18 台摄像机，>100 万张图像，覆盖 ±80° 极端头姿和 ±80° 极端视线）

### 性能指标

| 数据集 | 协议 | 平均角度误差 |
|--------|------|------------|
| ETH-XGaze | within-dataset | ~5.0°（需提交官方 leaderboard 获取） |
| MPIIFaceGaze | cross-dataset | ~5.5–6.0°（迁移性） |

### 优点

- ✅ **极端头姿鲁棒**：数据集覆盖 ±80° 头姿，最贴近驾驶场景（驾驶员会剧烈转头）
- ✅ 数据归一化可有效消除头姿影响，从根本上改善精度
- ✅ 数据集体量最大（>100 万样本），训练出的模型泛化更好
- ✅ 18 台摄像机采集，模拟多相机车内场景
- ✅ 提供预训练模型和 demo 代码

### 缺点

- ❌ **CC BY-NC-SA 4.0，不可商用**
- ❌ 数据集申请流程繁琐（需申请并下载>100GB 数据）
- ❌ 代码依赖较旧（Python 3.5, PyTorch 1.1），需适配
- ❌ 需要相机标定文件（`cam*.xml`）才能运行完整 demo
- ❌ 仅提供基础 baseline 训练代码，无开箱即用推理 Pipeline

### 驾驶员监控适用性

**✅ 高**（技术层面，License 限制须关注）。ETH-XGaze 数据集的极端头姿覆盖对驾驶场景最为关键。数据归一化方法论值得借鉴并集成到其他方案中。

---

## 六、pytorch_mpiigaze

### 基本信息

- **GitHub URL**：https://github.com/hysts/pytorch_mpiigaze
- **Stars / Forks**：⭐ 379（截至 2026-05-06）
- **License**：MIT
- **描述**：MPIIGaze 和 MPIIFaceGaze 的非官方 PyTorch 实现
- **依赖**：Python, PyTorch, OpenCV, OmegaConf

### 方法与原理

实现了 MPIIGaze 系列论文中的标准外观回归方法：

- **MPIIGaze**：眼部图像（60×36）→ LeNet 改进 CNN → pitch/yaw，需要头姿作为辅助输入
- **MPIIFaceGaze**：整脸图像（224×224）→ VGG-16 特征提取 → pitch/yaw，直接从人脸图像回归

支持通过 OmegaConf 进行配置化训练和评估，完整实现了 leave-one-person-out 交叉验证协议。

### 性能指标

| 数据集 | 方法 | 误差 |
|--------|------|------|
| MPIIGaze | LeNet-based | ~5.5° |
| MPIIFaceGaze | VGG-16 | ~4.5–5.0° |

### 优点

- ✅ MIT 协议，可商用
- ✅ 代码规范、实现清晰，适合研究和学习
- ✅ 完整支持 MPII 数据集的训练和评估

### 缺点

- ❌ 无完整实时推理 Pipeline（无内置人脸检测）
- ❌ 精度是业界中等水平（不是 SOTA）
- ❌ MPIIGaze 数据集场景（室内、桌面）与驾驶场景差异较大

### 驾驶员监控适用性

**⚠️ 中**。适合作为研究基线，了解外观基方法。不建议直接用于驾驶场景，需要封装推理 Pipeline 并微调。

---

## 七、GazeML

### 基本信息

- **GitHub URL**：https://github.com/swook/GazeML
- **Stars / Forks**：⭐ 579（截至 2026-04-20）
- **License**：MIT
- **框架**：TensorFlow 1.x（已过时）
- **依赖**：TensorFlow 1.x, dlib, OpenCV

### 方法与原理

**ELG（Eye Landmark and Gaze）模型**：

1. 检测眼部区域（dlib 关键点）
2. CNN 先预测眼部轮廓/虹膜关键点（Landmark Heatmap）
3. 再从特征图回归视线角度（2D gaze vector）
4. 其他可选模型：`DPG`（Dense Person Gaze）

### 性能指标

| 数据集 | 误差 |
|--------|------|
| MPIIGaze | ~5.6° |

### 优点

- ✅ MIT 协议，可商用
- ✅ 关键点中间结果可解释性好

### 缺点

- ❌ **TensorFlow 1.x**，已严重过时
- ❌ 无完整人脸级输入支持
- ❌ 不支持 PyTorch
- ❌ 在极端条件（头姿、光照）下鲁棒性一般

### 驾驶员监控适用性

**❌ 低**。技术框架过时，不建议用于新项目。

---

## 八、综合评估与驾驶员监控推荐

### 技术选型建议

**场景一：研究原型 / 学术验证（快速出结果）**
> **推荐：L2CS-Net**
> - 开箱即用，一键 pip 安装，内置人脸检测，输出 pitch/yaw
> - 预训练模型效果良好（~4°误差）
> - 代码活跃维护

**场景二：商业产品 / 量产（需 MIT 或宽松协议）**
> **推荐：L2CS-Net + ETH-XGaze 方法论（数据归一化）**
> - L2CS-Net：MIT-like，可商用
> - 参考 ETH-XGaze 的数据归一化方案以提升极端头姿鲁棒性
> - 可收集车内数据进行 fine-tuning

**场景三：眼镜用户鲁棒性要求高**
> **推荐：RT-GENE（非商业研究）**
> - 唯一内置眼部 Inpainting 的方案
> - 注意 CC BY-NC-SA 协议限制

**场景四：从零训练大规模车内数据**
> **参考：ETH-XGaze 数据归一化方法 + L2CS-Net 损失函数**
> - 数据归一化去除头姿影响（使模型更易学习）
> - L2CS 分类+回归混合损失训练更稳定

### 关键技术要点（驾驶场景特殊要求）

| 需求 | 建议 |
|------|------|
| 极端头姿（转头、低头）鲁棒性 | 必须使用数据归一化（ETH-XGaze 方法）+ 大角度覆盖数据集 |
| 实时性（≥25fps） | L2CS-Net + GPU；或 GazeTracking（CPU，精度较低） |
| 眼镜处理 | RT-GENE inpainting；或 GAN 数据增强包含眼镜样本 |
| 多人（驾驶员+副驾）检测 | L2CS-Net（RetinaFace 支持多人） |
| 不依赖云端（边缘部署） | L2CS-Net / ETH-XGaze 模型均支持本地推理 |
| 夜间/近红外摄像头适配 | 所有模型均需使用近红外数据 fine-tuning |

### 视线角度误差对比可视化

```
GazeTracking    [无量化输出]
GazeML          ████████████████████████████░  5.6°
RT-GENE         ████████████████████████░░░░░  6.3°（集成）
ETH-XGaze       ████████████████████░░░░░░░░░  ~5.0°
pytorch_mpiigaze ████████████████████░░░░░░░░░  ~4.5–5.0°
L2CS-Net        ████████████████░░░░░░░░░░░░░  ~4.0–4.2°  ← 最佳
```

---

*本调研基于 GitHub 源码分析、官方论文和 README 文档整理，数据截至 2026-05-08。*

