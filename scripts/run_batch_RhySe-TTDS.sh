#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
PROJECT_ROOT="/home/user/ly/F5-TTDS"
DATA_ROOT="${PROJECT_ROOT}/data"

SPEAKER_GLOB="torgo_[fm][0-9][0-9]_pinyin"
# SPEAKER_GLOB="torgo_f01_pinyin"  # debug one speaker if needed

METADATA_NAME="metadata.csv"
AUDIO_SUBDIR="wavs"

# Full F5-TTDS uses the model trained with severity-aware / Chained-CFG structure.
CKPT_FILE="${PROJECT_ROOT}/ckpts/torgo_all_sev_4gpu/model_last.pt"
VOCAB_FILE="${PROJECT_ROOT}/data/torgo_all_pinyin/vocab.txt"

OUT_DIR="${PROJECT_ROOT}/exp/f5ttds_chained_cfg_rhyss_eval"

MODEL_NAME="F5TTS_v1_Base"
VOCODER_NAME="vocos"
LOAD_VOCODER_FROM_LOCAL=0

DEVICE="cuda:0"
USE_EMA=0

NFE_STEP=32
CFG_STRENGTH=2.0
SWAY_SAMPLING_COEF=-1.0
SPEED=1.0
REMOVE_SILENCE=0

REF_STRATEGY="longest_text"
EXCLUDE_REF_FROM_GENERATION=1

# Full F5-TTDS = Chained CFG + RhySS
SEVERITY_CONDITION_MODE="speaker"
CHAINED_CFG=1
TEXT_CFG_STRENGTH=1.5
SPEAKER_CFG_STRENGTH=1.0
SEVERITY_CFG_STRENGTH=0.5

ADAPTIVE_SWAY_MODE="rhythm"
SEVERITY_ID_TO_LABEL="0:low,1:moderate,2:high"
LOW_SWAY_COEF=-0.70
MODERATE_SWAY_COEF=-0.85
HIGH_SWAY_COEF=-0.95
RHYTHM_SWAY_GAMMA=0.20
RHYTHM_TARGET_WPS=2.5
RHYTHM_MAX_ENERGY_DELTA_DB=10.0
RHYTHM_MAX_MEAN_PAUSE_SEC=1.0

# Debug first with MAX_ITEMS=8. After confirming quality, set MAX_ITEMS=-1.
# MAX_ITEMS=8
MAX_ITEMS=-1
SKIP_EXISTING=1
OVERWRITE_EXISTING=0
ENABLE_SIM=1

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python batch_f5ttds_chained_cfg_rhyss_eval.py \
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
  --severity_condition_mode "${SEVERITY_CONDITION_MODE}" \
  --chained_cfg "${CHAINED_CFG}" \
  --text_cfg_strength "${TEXT_CFG_STRENGTH}" \
  --speaker_cfg_strength "${SPEAKER_CFG_STRENGTH}" \
  --severity_cfg_strength "${SEVERITY_CFG_STRENGTH}" \
  --adaptive_sway_mode "${ADAPTIVE_SWAY_MODE}" \
  --severity_id_to_label "${SEVERITY_ID_TO_LABEL}" \
  --low_sway_coef "${LOW_SWAY_COEF}" \
  --moderate_sway_coef "${MODERATE_SWAY_COEF}" \
  --high_sway_coef "${HIGH_SWAY_COEF}" \
  --rhythm_sway_gamma "${RHYTHM_SWAY_GAMMA}" \
  --rhythm_target_wps "${RHYTHM_TARGET_WPS}" \
  --rhythm_max_energy_delta_db "${RHYTHM_MAX_ENERGY_DELTA_DB}" \
  --rhythm_max_mean_pause_sec "${RHYTHM_MAX_MEAN_PAUSE_SEC}" \
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
echo "RhySS auxiliary log: ${OUT_DIR}/rhyss_aux_generation_log.csv"
