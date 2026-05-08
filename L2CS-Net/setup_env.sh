#!/usr/bin/env bash
# =============================================================================
# L2CS-Net 一键环境配置脚本
# 适用：AutoDL / Ubuntu 20.04+ / Python 3.8+
# 用法：bash setup_env.sh
# =============================================================================

set -e  # 任何命令失败立即退出

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; \
              echo -e "${CYAN}  $*${NC}"; \
              echo -e "${CYAN}══════════════════════════════════════════${NC}"; }

# ─────────────────────────────────────────────────────────────────────────────
# 0. 定位脚本所在目录（L2CS-Net 根目录）
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log_info "Working directory: $SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 1. 检测 Python
# ─────────────────────────────────────────────────────────────────────────────
log_step "1/6  检测 Python 环境"

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    log_error "未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python: $($PYTHON --version)  -> $PYTHON"

# 版本检查：要求 >= 3.8
if $PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)"; then
    log_info "Python 版本满足要求 (>= 3.8)"
else
    log_error "Python $PY_VER 过旧，需要 3.8+"
    exit 1
fi

PIP="$PYTHON -m pip"

# ─────────────────────────────────────────────────────────────────────────────
# 2. 升级 pip / setuptools / wheel
# ─────────────────────────────────────────────────────────────────────────────
log_step "2/6  升级 pip / setuptools / wheel"

$PIP install --upgrade pip setuptools wheel -q
log_info "pip 升级完成: $($PIP --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. 安装 PyTorch（检查是否已安装，避免重复下载）
# ─────────────────────────────────────────────────────────────────────────────
log_step "3/6  检测并安装 PyTorch"

check_torch() {
    $PYTHON -c "import torch; print('torch', torch.__version__)" 2>/dev/null
}

if check_torch; then
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    log_info "PyTorch 已安装: $TORCH_VER，跳过安装"
else
    log_info "PyTorch 未安装，开始安装..."

    # 检测 CUDA 可用性
    CUDA_VER=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" | head -1)
        log_info "检测到 NVIDIA GPU，CUDA 版本: $CUDA_VER"
    else
        log_warn "未检测到 nvidia-smi，将安装 CPU 版 PyTorch"
    fi

    # 选择合适的 PyTorch wheel
    if [[ -n "$CUDA_VER" ]]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        CUDA_VER_INT=$((CUDA_MAJOR * 10 + CUDA_MINOR))

        if [[ $CUDA_VER_INT -ge 121 ]]; then
            TORCH_IDX="https://download.pytorch.org/whl/cu121"
            log_info "使用 CUDA 12.1 PyTorch wheel"
        elif [[ $CUDA_VER_INT -ge 118 ]]; then
            TORCH_IDX="https://download.pytorch.org/whl/cu118"
            log_info "使用 CUDA 11.8 PyTorch wheel"
        else
            TORCH_IDX="https://download.pytorch.org/whl/cu117"
            log_info "使用 CUDA 11.7 PyTorch wheel"
        fi
    else
        TORCH_IDX="https://download.pytorch.org/whl/cpu"
        log_info "使用 CPU PyTorch wheel"
    fi

    $PIP install torch torchvision --index-url "$TORCH_IDX" -q
    if check_torch; then
        log_info "PyTorch 安装成功: $($PYTHON -c 'import torch; print(torch.__version__)')"
    else
        log_error "PyTorch 安装失败"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. 安装 L2CS-Net 依赖
# ─────────────────────────────────────────────────────────────────────────────
log_step "4/6  安装 L2CS-Net 依赖包"

# 逐个检测并安装，避免重复安装
install_if_missing() {
    local package="$1"
    local import_name="${2:-$1}"
    if $PYTHON -c "import $import_name" 2>/dev/null; then
        log_info "$package 已安装，跳过"
    else
        log_info "安装 $package ..."
        $PIP install "$package" -q
    fi
}

install_if_missing "matplotlib"
install_if_missing "numpy"
install_if_missing "opencv-python" "cv2"
install_if_missing "pandas"
install_if_missing "Pillow" "PIL"
install_if_missing "scipy"
install_if_missing "gdown"

# face_detection（椭圆脸检测，依赖 git+url）
if $PYTHON -c "import face_detection" 2>/dev/null; then
    log_info "face_detection 已安装，跳过"
else
    log_info "安装 face_detection ..."
    $PIP install git+https://github.com/elliottzheng/face-detection -q
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. 安装 L2CS-Net 本体（editable install）
# ─────────────────────────────────────────────────────────────────────────────
log_step "5/6  安装 L2CS-Net 包（editable）"

if $PYTHON -c "import l2cs" 2>/dev/null; then
    log_info "l2cs 已安装，跳过"
else
    log_info "以可编辑模式安装 l2cs ..."
    $PIP install -e . -q
fi

# 验证
if $PYTHON -c "import l2cs; print('l2cs OK')" 2>/dev/null; then
    log_info "l2cs 导入验证通过"
else
    log_error "l2cs 导入失败，请检查安装"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. 下载预训练模型
# ─────────────────────────────────────────────────────────────────────────────
log_step "6/6  下载预训练模型"

MODEL_PATH="$SCRIPT_DIR/models/L2CSNet_gaze360.pkl"
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/output"

# 设置 HuggingFace 镜像（AutoDL / 国内环境）
export HF_ENDPOINT="https://hf-mirror.com"
log_info "HuggingFace endpoint: $HF_ENDPOINT"

if [[ -f "$MODEL_PATH" ]]; then
    SIZE_MB=$(du -m "$MODEL_PATH" | cut -f1)
    if [[ $SIZE_MB -ge 50 ]]; then
        log_info "模型已存在且大小正常 (${SIZE_MB}MB)，跳过下载"
    else
        log_warn "模型文件过小 (${SIZE_MB}MB)，重新下载..."
        rm -f "$MODEL_PATH"
        $PYTHON "$SCRIPT_DIR/download_model.py"
    fi
else
    $PYTHON "$SCRIPT_DIR/download_model.py"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────────
echo ""
log_info "══════════════════════════════════════════"
log_info "  环境配置完成！"
log_info "══════════════════════════════════════════"
echo ""
log_info "验证安装："
$PYTHON -c "
import torch, cv2, numpy, l2cs
print(f'  torch       : {torch.__version__}')
print(f'  CUDA 可用   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 数量    : {torch.cuda.device_count()}')
    print(f'  GPU 名称    : {torch.cuda.get_device_name(0)}')
print(f'  opencv       : {cv2.__version__}')
print(f'  numpy        : {numpy.__version__}')
print(f'  l2cs        : OK')
"
echo ""
log_info "快速测试命令："
echo "  python run_demo.py --input datasets/result_long.mp4 --device cpu --output output/result_cpu.mp4"
echo ""
log_info "GPU 测试命令（有 GPU 时）："
echo "  python run_demo.py --input datasets/result_long.mp4 --device gpu:0 --output output/result_gpu.mp4"
echo ""
