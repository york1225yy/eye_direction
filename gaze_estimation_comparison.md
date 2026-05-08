# 驾驶员视线估计开源算法对比文档

> 更新日期：2026-05-08  
> 场景定位：车辆驾驶员注视方向检测（Driver Gaze Estimation / Eye Direction Monitoring）  
> 筛选条件：Python 实现、优先 PyTorch 框架、具备可量化角度输出或可集成至驾驶监控管线

---

## 一、候选项目速览

| # | 项目 | GitHub | 框架 | 输出类型 | 开源协议 | Stars（参考） |
|---|------|--------|------|----------|----------|--------------|
| 1 | **L2CS-Net** | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | PyTorch | Pitch / Yaw 角度 | MIT ✅ | ~1.5k |
| 2 | **GazeTracking** | [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking) | OpenCV + dlib | 左/中/右离散分类 | MIT ✅ | ~3k |
| 3 | **ETH-XGaze** | [xucong-zhang/ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) | PyTorch | Pitch / Yaw 角度 | CC BY-NC-SA ⚠️ | ~600 |
| 4 | **RT-GENE** | [Tobias-Fischer/rt_gene](https://github.com/Tobias-Fischer/rt_gene) | PyTorch / TF | Pitch / Yaw 角度 | CC BY-NC-SA ⚠️ | ~600 |
| 5 | **pytorch_mpiigaze** | [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze) | PyTorch | Pitch / Yaw 角度 | MIT ✅ | ~800 |
| 6 | **GazeML** | [swook/GazeML](https://github.com/swook/GazeML) | TensorFlow 1.x | 瞳孔向量 / 角度 | MIT ✅ | ~1.5k |

---

## 二、各项目详细分析

### 2.1 GazeTracking（用户指定）

- **项目地址**：https://github.com/antoinelame/GazeTracking
- **方法原理**：纯传统计算机视觉方案。使用 dlib 的 68 点人脸关键点检测提取眼部 ROI，通过瞳孔在眼眶内的位置比例（Horizontal Ratio / Vertical Ratio）判断注视方向。
- **框架依赖**：OpenCV、dlib（**非 PyTorch**）
- **训练数据集**：无模型训练，基于规则计算
- **性能指标**：无量化角度误差，仅输出"向左 / 居中 / 向右"离散状态
- **实时性**：CPU 可实时运行，速度快
- **驾驶场景适用性**：⚠️ **低**
  - 无法输出准确的注视角度（Pitch/Yaw），不满足 DMS（Driver Monitoring System）通常要求的偏转角度阈值判断
  - 对头部姿态变化（点头、侧头）不鲁棒，驾驶员场景中头姿变化频繁
  - 无法区分"向前方直视但眼球偏转"与"头部偏转"
- **优点**：实现极简、依赖少、无 GPU 要求、易于集成原型验证
- **缺点**：精度低、无角度输出、不适合量化 DMS 标准、项目已停止维护（最后更新 ~2019）

---

### 2.2 L2CS-Net（用户指定，**推荐**）

- **项目地址**：https://github.com/Ahmednull/L2CS-Net
- **论文**：*L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments* (2022)
- **方法原理**：
  - 骨干网络：**ResNet-50**（预训练于 ImageNet）
  - 将注视方向估计分解为分类（Classification）+ 回归（Regression）联合任务（Combined Loss = CLS Loss + MSE Loss），分别预测 Pitch 和 Yaw 的离散 bins 及偏移量
  - 人脸检测：内置 RetinaFace 检测器，自动裁剪人脸 patch
  - 输入：224×224 人脸图像
  - 输出：连续的 pitch（俯仰）、yaw（偏航）弧度值
- **框架依赖**：**PyTorch**、torchvision、RetinaFace
- **训练数据集**：
  - Gaze360（~172k 图像，室内+室外，±180° 全向注视）
  - MPIIFaceGaze（~45k 图像，笔记本摄像头）
- **性能指标**（Angular Error，越低越好）：

  | 数据集 | Angular Error |
  |--------|--------------|
  | Gaze360 | **~3.92°** |
  | MPIIFaceGaze | **~4.20°** |

- **实时性**：GPU 下约 30+ FPS（ResNet-50 推理）；可换轻量主干提速
- **驾驶场景适用性**：✅ **高**
  - 输出连续 pitch/yaw 角度，可直接对接"视线偏离前方超过 X° 持续 Y 秒"的 DMS 规则
  - Gaze360 训练集覆盖多种头部姿态，泛化能力较强
  - MIT 协议，可用于商业产品
- **优点**：精度最优（候选中 ~4°）、PyTorch 原生、内置完整检测管线、MIT 协议、代码结构清晰
- **缺点**：ResNet-50 在嵌入式端（如车载 SoC）需量化/剪枝；对遮挡（墨镜、口罩）无专项处理；Gaze360 场景与车内环境有域差

---

### 2.3 ETH-XGaze

- **项目地址**：https://github.com/xucong-zhang/ETH-XGaze
- **论文**：*ETH-XGaze: A Large Scale Dataset and Benchmark for Gaze Estimation under Extreme Head Poses and Gaze Directions* (ECCV 2020)
- **方法原理**：
  - 骨干网络：**ResNet-50**
  - 核心技术：**凝视归一化（Gaze Normalization）**——将不同摄像头外参、不同头部姿态下的人脸图像统一变换到标准视图，消除头姿对注视估计的干扰
  - 数据集特色：极端头姿（±80° pitch/yaw）、多摄像头、高分辨率（6MP）多视角标注
- **框架依赖**：**PyTorch**
- **训练数据集**：ETH-XGaze（~1,083k 图像，110 名受试者，18 个摄像机）
- **性能指标**：

  | 数据集 | Angular Error |
  |--------|--------------|
  | ETH-XGaze (normalized) | **~5.0°** |
  | MPIIFaceGaze (cross-dataset) | ~5.5° |

- **驾驶场景适用性**：✅ **高（头姿鲁棒性最佳）**
  - ±80° 头姿覆盖了驾驶员侧头查看后视镜、低头看仪表盘等极端姿态
  - 凝视归一化方法是 DMS 系统最值得借鉴的技术方案
  - 可将该归一化作为预处理模块移植到其他模型
- **优点**：头姿鲁棒性最强、数据归一化方法工程价值高、大规模数据集
- **缺点**：开源协议为 **CC BY-NC-SA 4.0（非商业）**，量产需洽谈授权；精度略低于 L2CS-Net；数据集申请需邮件联系

---

### 2.4 RT-GENE

- **项目地址**：https://github.com/Tobias-Fischer/rt_gene
- **论文**：*RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments* (ECCV 2018)
- **方法原理**：
  - 输入：**双眼 patch + 头部姿态**（6DoF）联合估计
  - 骨干：多个 VGG-16 / ResNet 特征提取，集成多模型预测（Ensemble）
  - 特色：内置 **眼镜去除（Inpainting）** 模块，对戴眼镜场景有专项优化
  - 输出：Pitch / Yaw 角度
- **框架依赖**：PyTorch（推理）+ TensorFlow（部分模块），依赖较重
- **训练数据集**：RT-GENE Dataset（~122k 标注帧，自然环境，含眼镜/无眼镜）
- **性能指标**：

  | 方法 | Angular Error |
  |------|--------------|
  | 集成模型（Ensemble） | ~6.3° |
  | 单模型 | ~7.1° |

- **驾驶场景适用性**：⚠️ **中**
  - 眼镜处理能力是候选中最强的，适合眼镜佩戴率高的场景
  - 集成模型推理速度慢，实时性较差
  - 精度弱于 L2CS-Net 和 ETH-XGaze
- **优点**：眼镜场景鲁棒性最好、双眼联合建模理论上更准确
- **缺点**：CC BY-NC-SA 协议、依赖复杂（TF+PyTorch 混合）、精度偏低、Ensemble 推理慢

---

### 2.5 pytorch_mpiigaze

- **项目地址**：https://github.com/hysts/pytorch_mpiigaze
- **方法原理**：
  - MPIIGaze 系列数据集上的 PyTorch 复现，包含多种骨干：
    - **LeNet**（轻量级，眼部图像输入）
    - **VGG-16**（人脸图像输入）
    - **ResNet-50**（人脸图像输入）
  - 同时支持 MPIIGaze（眼图）和 MPIIFaceGaze（脸图）两种输入模式
  - 也提供了 Gaze360 的训练脚本
- **框架依赖**：**PyTorch**（纯 PyTorch，依赖干净）
- **训练数据集**：MPIIGaze、MPIIFaceGaze、Gaze360
- **性能指标**：

  | 模型 | 数据集 | Angular Error |
  |------|--------|--------------|
  | ResNet-50 | MPIIFaceGaze | ~4.5° |
  | VGG-16 | MPIIFaceGaze | ~5.0° |
  | LeNet | MPIIGaze | ~5.5° |

- **驾驶场景适用性**：⚠️ **中**
  - MPIIGaze 采集自笔记本摄像头（用户坐姿，视角与驾驶舱类似）
  - 代码结构良好，便于自定义训练和迁移学习
  - 无内置端到端检测管线，需自行集成人脸/眼部检测器
- **优点**：代码质量高、模块化、MIT 协议、支持多骨干切换、便于自训练
- **缺点**：MPIIFaceGaze 场景多样性有限；无内置检测流水线；精度不及 L2CS-Net

---

### 2.6 GazeML

- **项目地址**：https://github.com/swook/GazeML
- **方法原理**：
  - 基于 **ELG（Eye Like Gazer）** 架构：沙漏网络（Hourglass Network）进行眼部关键点检测（瞳孔、虹膜轮廓），再从关键点推算注视向量
  - 额外模块：DPG（Dense Probabilistic Gaze）实现不确定性估计
- **框架依赖**：**TensorFlow 1.x**（非 PyTorch，且依赖已停止维护的 TF1）
- **性能指标**：~5.6°（MPIIGaze）
- **驾驶场景适用性**：❌ **低**
  - TF1 依赖导致部署困难，PyTorch 生态不兼容
  - 精度中等，无头姿补偿
  - 项目已多年未更新
- **优点**：关键点可解释性强、提供不确定性估计
- **缺点**：TF1 框架已过时、不兼容 PyTorch 生态、维护停滞

---

## 三、综合对比表

| 评估维度 | GazeTracking | **L2CS-Net** | ETH-XGaze | RT-GENE | pytorch_mpiigaze | GazeML |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **框架** | OpenCV/dlib | PyTorch ✅ | PyTorch ✅ | PyTorch+TF | PyTorch ✅ | TF1 ❌ |
| **角度误差** | N/A | **~4.0°** ⭐ | ~5.0° | ~6.3° | ~4.5° | ~5.6° |
| **输出格式** | 离散分类 | Pitch/Yaw ✅ | Pitch/Yaw ✅ | Pitch/Yaw ✅ | Pitch/Yaw ✅ | 向量 |
| **头姿鲁棒性** | ❌ 差 | ✅ 中 | ✅✅ 最强 | ✅ 中 | ✅ 中 | ❌ 差 |
| **眼镜鲁棒性** | ❌ 差 | ⚠️ 一般 | ⚠️ 一般 | ✅✅ 最强 | ⚠️ 一般 | ⚠️ 一般 |
| **内置检测管线** | ✅ | ✅ | ⚠️ 部分 | ✅ | ❌ | ❌ |
| **实时性（GPU）** | CPU ✅ | ~30 FPS ✅ | ~25 FPS ✅ | 慢（集成）⚠️ | ~30 FPS ✅ | 慢 ⚠️ |
| **商业协议** | MIT ✅ | **MIT ✅** | CC BY-NC-SA ⚠️ | CC BY-NC-SA ⚠️ | MIT ✅ | MIT ✅ |
| **驾驶场景适用** | ⚠️ 低 | ✅✅ **高** | ✅✅ **高** | ⚠️ 中 | ⚠️ 中 | ❌ 低 |
| **维护活跃度** | ❌ 停更 | ✅ 活跃 | ⚠️ 一般 | ⚠️ 一般 | ✅ 活跃 | ❌ 停更 |
| **易用性/文档** | ✅✅ 最佳 | ✅ 好 | ⚠️ 一般 | ⚠️ 一般 | ✅ 好 | ⚠️ 一般 |

---

## 四、驾驶监控场景建议

### 4.1 推荐方案排序

```
第一优先：L2CS-Net
├── 精度最优（~4°）
├── 完整端到端 PyTorch 管线
├── MIT 协议（可商用）
└── 开箱即用，适合快速原型和量产

第二优先：ETH-XGaze（技术方案参考）
├── 数据归一化方法最适合驾驶场景（极端头姿）
├── 建议将其 gaze normalization 模块移植到 L2CS-Net 的预处理
└── 注意：非商业协议，量产需授权

补充方案：pytorch_mpiigaze
├── 代码模块化好，适合自定义迁移学习
└── 可用驾驶场景数据 fine-tune L2CS-Net 的替代训练框架
```

### 4.2 推荐技术路线（实际落地）

```
摄像头输入（IR 或 RGB）
       │
       ▼
人脸检测（RetinaFace / YOLOv8-face）
       │
       ▼
凝视归一化（借鉴 ETH-XGaze 方法）
       │
       ▼
L2CS-Net（ResNet-50 → 轻量化可换 ResNet-18/MobileNetV3）
       │
       ▼
Pitch / Yaw 角度输出
       │
       ▼
DMS 规则引擎（如：yaw > 30° 持续 > 2s → 分心报警）
```

### 4.3 工程注意事项

| 注意点 | 说明 |
|--------|------|
| **域适应** | 开源模型均在通用数据集训练，建议收集车内实际数据进行 Fine-tune |
| **光照适应** | 车内逆光/夜间弱光需配合 IR 摄像头或图像增强 |
| **眼镜处理** | 如驾驶员佩戴墨镜，建议在训练集中加入含墨镜样本 |
| **头姿解耦** | 建议使用 ETH-XGaze 的归一化方法，将头姿变换从注视估计中解耦 |
| **嵌入式部署** | ResNet-50 → ResNet-18 或 MobileNetV3 可在车载 SoC（如 TDA4/Orin）上实时运行 |
| **时序平滑** | 对角度输出加 Kalman 滤波或滑动平均，减少抖动 |

---

## 五、参考资料

- L2CS-Net 论文：[arXiv:2203.03339](https://arxiv.org/abs/2203.03339)
- ETH-XGaze 论文：[ECCV 2020](https://arxiv.org/abs/2007.15837)
- RT-GENE 论文：[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.pdf)
- MPIIFaceGaze 数据集：[MPI Informatics](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Gaze360 数据集：[MIT CSAIL](http://gaze360.csail.mit.edu/)
- ETH-XGaze 数据集申请：[ETH Zurich](https://ait.ethz.ch/xgaze)
