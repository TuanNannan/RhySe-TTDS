#!/usr/bin/env bash
set -euo pipefail

# Batch F5-TTS TORGO generation/evaluation with direct inference and USE_EMA=0.
# Put this file and batch_f5tts_torgo_direct_use_ema0_generate_eval.py under:
#   /home/user/ly/F5-TTS-main/
# Run:
#   cd /home/user/ly/F5-TTS-main
#   bash run_batch_f5tts_torgo_direct_use_ema0_generate_eval.sh

# =========================
# [PATH-CHECK-1] F5-TTS repository root
# =========================
PROJECT_ROOT="/home/user/ly/F5-TTS-main"

# =========================
# [PATH-CHECK-2] Data root containing folders like:
#   data/torgo_f01_pinyin/metadata.csv
#   data/torgo_f03_pinyin/metadata.csv
#   data/torgo_m01_pinyin/metadata.csv
# =========================
DATA_ROOT="${PROJECT_ROOT}/data"
SPEAKER_GLOB="torgo_*_pinyin"
METADATA_NAME="metadata.csv"

# =========================
# [PATH-CHECK-3] Audio location.
# If wav files are directly under each speaker folder, keep empty.
# If wav files are under torgo_xxx_pinyin/wavs, set AUDIO_SUBDIR="wavs".
# =========================
AUDIO_SUBDIR="wavs"

# =========================
# [PATH-CHECK-4] Fine-tuned checkpoint root and pattern.
# Your current checkpoint example:
#   /home/user/ly/F5-TTS-main/ckpts/torgo_f01/model_100.pt
# Recommended after debugging:
#   /home/user/ly/F5-TTS-main/ckpts/torgo_f01/model_last.pt
# The script maps data folder torgo_f01_pinyin -> ckpt folder torgo_f01.
# =========================
CKPT_ROOT="${PROJECT_ROOT}/ckpts"
CKPT_PATTERN="{ckpt_dataset}/model_last.pt"
# If you intentionally want model_100.pt, use this instead:
# CKPT_PATTERN="{ckpt_dataset}/model_100.pt"

# =========================
# [PATH-CHECK-5] Custom vocab.
# Keep VOCAB_FILE empty and AUTO_VOCAB=1 to automatically try:
#   data/torgo_f01_pinyin/vocab.txt
#   ckpts/torgo_f01/vocab.txt
# =========================
VOCAB_FILE=""
VOCAB_PATTERN=""
AUTO_VOCAB=1

# =========================
# [PATH-CHECK-6] Output directory.
# Change this to a new directory if you want to avoid reusing old bad wavs.
# =========================
OUT_DIR="${PROJECT_ROOT}/exp/f5ttds_torgo_severity_adaptive_ss_eval"

# =========================
# [INFER-CHECK-1] Model architecture name.
# This MUST match your fine-tuning base model.
# The minimal script worked with your current setting; keep it the same here.
# If needed, switch between F5TTS_v1_Base and F5TTS_Base.
# =========================
MODEL_NAME="F5TTS_v1_Base"
# MODEL_NAME="F5TTS_Base"

# =========================
# [INFER-CHECK-2] EMA setting.
# You confirmed USE_EMA=0 sounds closer to the fine-tuned original voice.
# =========================
USE_EMA=0

# =========================
# [INFER-CHECK-3] Generation settings.
# Do not enable REMOVE_SILENCE for dysarthric rhythm evaluation.
# =========================
NFE_STEP=32
CFG_STRENGTH=2.0
SWAY_SAMPLING_COEF=-1.0
# fixed: original F5-TTS; severity: F5-TTDS severity-aware adaptive Sway Sampling
ADAPTIVE_SWAY_MODE="severity"
SPEED=1.0
VOCODER_NAME="vocos"
REMOVE_SILENCE=0

# =========================
# [RUN-CHECK-1] Reference selection.
# longest_text: one fixed long prompt per speaker; recommended first.
# self: each utterance uses itself as reference, not recommended for augmentation but useful for sanity checks.
# =========================
REF_STRATEGY="longest_text"
EXCLUDE_REF_FROM_GENERATION=1

# =========================
# [RUN-CHECK-2] Debug or full run.
# First run with MAX_ITEMS=6 and listen to generated_wavs.
# After confirming quality, change MAX_ITEMS=-1.
# =========================
MAX_ITEMS=-1
# MAX_ITEMS=-1
SKIP_EXISTING=1
OVERWRITE_EXISTING=0

# Speaker similarity can require downloading SpeechBrain model.
# If network/model download is a problem, set ENABLE_SIM=0 first.
ENABLE_SIM=1

cd "${PROJECT_ROOT}"

python batch_f5tts_eval2_adaptive_ss.py \
  --project_root "${PROJECT_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --speaker_glob "${SPEAKER_GLOB}" \
  --metadata_name "${METADATA_NAME}" \
  --audio_subdir "${AUDIO_SUBDIR}" \
  --ref_strategy "${REF_STRATEGY}" \
  --exclude_ref_from_generation "${EXCLUDE_REF_FROM_GENERATION}" \
  --ckpt_root "${CKPT_ROOT}" \
  --ckpt_pattern "${CKPT_PATTERN}" \
  --vocab_file "${VOCAB_FILE}" \
  --vocab_pattern "${VOCAB_PATTERN}" \
  --auto_vocab "${AUTO_VOCAB}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --vocoder_name "${VOCODER_NAME}" \
  --device cuda \
  --use_ema "${USE_EMA}" \
  --nfe_step "${NFE_STEP}" \
  --cfg_strength "${CFG_STRENGTH}" \
  --sway_sampling_coef "${SWAY_SAMPLING_COEF}" \
  --adaptive_sway_mode "${ADAPTIVE_SWAY_MODE}" \
  --speed "${SPEED}" \
  --remove_silence "${REMOVE_SILENCE}" \
  --skip_existing "${SKIP_EXISTING}" \
  --overwrite_existing "${OVERWRITE_EXISTING}" \
  --max_items "${MAX_ITEMS}" \
  --enable_sim "${ENABLE_SIM}"

echo "Done. Check outputs under: ${OUT_DIR}"
echo "First listen to: ${OUT_DIR}/generated_wavs/"
echo "Then inspect: ${OUT_DIR}/per_utterance_metrics.csv"
