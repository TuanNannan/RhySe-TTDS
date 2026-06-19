#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/user/ly/F5-TTS-main"

DATA_ROOT="${PROJECT_ROOT}/data"
SPEAKER_GLOB="torgo_[fm][0-9][0-9]_pinyin"
METADATA_NAME="metadata.csv"
AUDIO_SUBDIR="wavs"

CKPT_FILE="${PROJECT_ROOT}/ckpts/torgo_all_sev_4gpu/model_last.pt"
VOCAB_FILE="${PROJECT_ROOT}/data/torgo_all_pinyin/vocab.txt"

# 这个输出目录建议每个消融实验单独改名
OUT_DIR="${PROJECT_ROOT}/exp/f5ttds_unified_sevcond_fixedsway_eval"

MODEL_NAME="F5TTS_v1_Base"
VOCODER_NAME="vocos"

# 如果本地有 checkpoints/vocos-mel-24khz，就设 1；没有就设 0，并建议开启 HF_ENDPOINT 镜像。
LOAD_VOCODER_FROM_LOCAL=0

DEVICE="cuda"
USE_EMA=0

NFE_STEP=32
CFG_STRENGTH=2.0

# fixed: 普通固定 Sway
# severity: 病理严重程度感知 Adaptive Sway
ADAPTIVE_SWAY_MODE="fixed"
SWAY_SAMPLING_COEF=-1.0

# none: 不传 severity，做消融
# speaker: 根据 speaker 的 severity label 传入 0/1/2
SEVERITY_CONDITION_MODE="speaker"

SPEED=1.0
REMOVE_SILENCE=0

REF_STRATEGY="longest_text"
EXCLUDE_REF_FROM_GENERATION=1

MAX_ITEMS=8
SKIP_EXISTING=1
OVERWRITE_EXISTING=0

ENABLE_SIM=1

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python batch_f5ttds_severity_unified_eval.py \
  --project_root "${PROJECT_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --speaker_glob "${SPEAKER_GLOB}" \
  --metadata_name "${METADATA_NAME}" \
  --audio_subdir "${AUDIO_SUBDIR}" \
  --ckpt_file "${CKPT_FILE}" \
  --vocab_file "${VOCAB_FILE}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --vocoder_name "${VOCODER_NAME}" \
  --load_vocoder_from_local "${LOAD_VOCODER_FROM_LOCAL}" \
  --device "${DEVICE}" \
  --use_ema "${USE_EMA}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
  --adaptive_sway_mode "${ADAPTIVE_SWAY_MODE}" \
  --severity_condition_mode "${SEVERITY_CONDITION_MODE}" \
  --speed "${SPEED}" \
  --remove_silence "${REMOVE_SILENCE}" \
  --ref_strategy "${REF_STRATEGY}" \
  --exclude_ref_from_generation "${EXCLUDE_REF_FROM_GENERATION}" \
  --skip_existing "${SKIP_EXISTING}" \
  --overwrite_existing "${OVERWRITE_EXISTING}" \
  --max_items "${MAX_ITEMS}" \
  --enable_sim "${ENABLE_SIM}"

echo "Done. Check outputs under: ${OUT_DIR}"
echo "Generated wavs: ${OUT_DIR}/generated_wavs/"
echo "Metrics: ${OUT_DIR}/per_utterance_metrics.csv"