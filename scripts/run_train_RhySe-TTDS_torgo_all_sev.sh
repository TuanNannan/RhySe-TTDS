#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/user/ly/F5-TTS-main"

DATASET_PROJECT="torgo_all_pinyin"
DATASET_NAME="torgo_all"
TOKENIZER="pinyin"

DATA_DIR="${PROJECT_ROOT}/data/${DATASET_PROJECT}"
CKPT_DIR="${PROJECT_ROOT}/ckpts/torgo_all_sev_4gpu"

LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train_torgo_all_sev_4gpu.log"

PRETRAIN_PATH="/home/user/ly/F5-TTS-main/ckpts/torgo_all_sev_4gpu/pretrained_model_1250000.safetensors"

# =========================
# 4×L40S 多卡设置
# =========================
GPU_IDS="0,1,2,3"
NUM_PROCESSES=4

# =========================
# Gradio baseline 参数
# =========================
EXP_NAME="F5TTS_v1_Base"

EPOCHS=50
LEARNING_RATE=0.00001
MAX_GRAD_NORM=1
NUM_WARMUP_UPDATES=50

BATCH_SIZE_TYPE="frame"

# 这是“每张 GPU”的 frame batch，不是总 batch。
# 4 卡时有效 batch 约为 1600 × 4。
BATCH_SIZE_PER_GPU=1600
GRAD_ACCUMULATION_STEPS=1
MAX_SAMPLES=10

SAVE_PER_UPDATES=500
KEEP_LAST_N_CHECKPOINTS=10
LAST_PER_UPDATES=100

MIXED_PRECISION="fp16"
LOGGER="tensorboard"

cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
# NCCL 设置，单机多卡通常更稳
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8

echo "============================================================"
echo "F5-TTDS severity-aware 4-GPU training"
echo "Project root      : ${PROJECT_ROOT}"
echo "Dataset dir       : ${DATA_DIR}"
echo "Checkpoint dir    : ${CKPT_DIR}"
echo "Visible GPUs      : ${GPU_IDS}"
echo "Num processes     : ${NUM_PROCESSES}"
echo "Batch per GPU     : ${BATCH_SIZE_PER_GPU}"
echo "Grad accum        : ${GRAD_ACCUMULATION_STEPS}"
echo "Mixed precision   : ${MIXED_PRECISION}"
echo "Log file          : ${LOG_FILE}"
echo "============================================================"

if [ ! -d "${DATA_DIR}" ]; then
  echo "[ERROR] Dataset directory not found: ${DATA_DIR}"
  exit 1
fi

if [ ! -f "${DATA_DIR}/duration.json" ]; then
  echo "[ERROR] duration.json not found: ${DATA_DIR}/duration.json"
  exit 1
fi

if [ ! -d "${DATA_DIR}/raw" ] && [ ! -f "${DATA_DIR}/raw.arrow" ]; then
  echo "[ERROR] raw dataset not found under: ${DATA_DIR}"
  exit 1
fi

python - <<PY
from pathlib import Path
from collections import Counter
from datasets import load_from_disk, Dataset

data_dir = Path("${DATA_DIR}")

if (data_dir / "raw").exists():
    ds = load_from_disk(str(data_dir / "raw"))
else:
    ds = Dataset.from_file(str(data_dir / "raw.arrow"))

print("num samples:", len(ds))
print("columns:", ds.column_names)

if "severity" not in ds.column_names:
    raise RuntimeError("severity column not found in dataset.")

print("severity counts:", Counter(ds["severity"]))
print("first sample:", ds[0])
PY

CMD=(
  accelerate launch
  --multi_gpu
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
  "src/f5_tts/train/finetune_cli.py"
  --exp_name "${EXP_NAME}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}"
  --batch_size_type "${BATCH_SIZE_TYPE}"
  --max_samples "${MAX_SAMPLES}"
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --epochs "${EPOCHS}"
  --num_warmup_updates "${NUM_WARMUP_UPDATES}"
  --save_per_updates "${SAVE_PER_UPDATES}"
  --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}"
  --last_per_updates "${LAST_PER_UPDATES}"
  --dataset_name "${DATASET_NAME}"
  --tokenizer "${TOKENIZER}"
  --finetune
  --checkpoint_path "${CKPT_DIR}"
  --logger "${LOGGER}"
  --log_samples
)

if [ -n "${PRETRAIN_PATH}" ]; then
  CMD+=(--pretrain "${PRETRAIN_PATH}")
fi

echo "============================================================"
echo "Training command:"
printf '%q ' "${CMD[@]}"
echo
echo "============================================================"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"