# RhySe-TTDS

**RhySe-TTDS: Rhythm-Aware and Severity-Disentangled Dysarthric Speech Synthesis for ASR Augmentation**

RhySe-TTDS is a dysarthric speech synthesis and ASR augmentation framework. Unlike conventional high-naturalness TTS, RhySe-TTDS is designed to generate dysarthric speech that preserves severity-related pathological acoustic and rhythm characteristics for downstream automatic speech recognition (ASR) augmentation.

The framework introduces two main components:

1. **Severity-Disentangled Chained Classifier-Free Guidance (Chained CFG)** — disentangles text, speaker, and severity guidance during inference to preserve severity-dependent pathological speech patterns while maintaining text fidelity and speaker conditioning.
2. **Rhythm-aware Adaptive Sway Sampling (RhySS)** — estimates utterance-level rhythm irregularity from reference speech and adapts the Sway Sampling coefficient accordingly.

Downstream ASR evaluation is conducted using Whisper, HuBERT, and wav2vec2 under TORGO-based leave-one-speaker-out (LOSO) evaluation settings.

---

## Repository Structure

```
RhySe-TTDS/
├── pyproject.toml                  # Python package metadata and CLI entry points
├── environment.yml                 # Conda environment specification
├── ruff.toml                       # Linter configuration
├── README.md
├── .gitignore
├── .gitattributes
│
├── src/rhyse_ttds/                 # Main RhySe-TTDS source code
│   ├── api.py                      # Public API
│   ├── socket_server.py            # Realtime socket server
│   ├── socket_client.py            # Socket client
│   ├── configs/                    # Model YAML configs (F5TTS_v1_Base, E2TTS, etc.)
│   ├── model/                      # CFM, DiT backbones, trainer, dataset utilities
│   ├── infer/                      # Inference CLI, Gradio UI, speech editing, utils
│   ├── train/                      # Fine-tuning CLI, Gradio UI, training loop
│   ├── eval/                       # Batch inference, evaluation metrics, LOSO eval shell script
│   ├── scripts/                    # Utility scripts (param counting, etc.)
│   └── runtime/triton_trtllm/      # Optional Triton/TensorRT-LLM deployment templates
│
├── scripts/                        # Training, generation, and evaluation orchestration
│   ├── run_train_RhySe-TTDS_torgo_all_sev.sh   # Multi-GPU training script
│   ├── run_batch_f5tts_eval2.sh                 # F5-TTS baseline generation
│   ├── run_batch_RhySe-TTDS.sh                  # RhySe-TTDS severity adaptive SS
│   ├── run_batch_RhySe-TTDS_eval2_adaptive_ss.sh
│   ├── run_batch_RhySe-TTDS_eval2_rhythm_adaptive_ss.sh   # RhySS generation
│   ├── run_batch_RhySe-TTDS_chained_cfg_eval.sh            # Chained CFG generation
│   ├── run_batch_RhySe-TTDS_severity_unified_eval.sh       # Severity-unified eval
│   ├── batch_f5tts_eval2.py                     # F5-TTS baseline batch eval
│   ├── batch_RhySe-TTDS_eval2_rhythm_adaptive_ss.py        # RhySS batch eval
│   ├── batch_RhySe-TTDS_chained_cfg_eval.py                 # Chained CFG batch eval
│   ├── batch_RhySe-TTDS_chained_cfg_rhyss_eval.py          # Full RhySe-TTDS batch eval
│   ├── batch_RhySe-TTDS_severity_unified_eval.py            # Unified severity eval base
│   ├── batch_RhySe-TTDS_eval2_adaptive_ss.py                # Severity adaptive SS eval
│   └── make_torgo_all_pinyin.py                 # Merge per-speaker data into unified set
│
├── data/                           # Data preparation scripts and per-speaker manifests
│   ├── prepare_torgo.py            # Extract and clean TORGO data into CSV
│   ├── prepare_torgo_dataset.py    # Per-speaker data preparation with pinyin conversion
│   ├── torgo_manifest_final.csv    # Master TORGO manifest (audio paths + transcripts)
│   ├── torgo_*_pinyin/             # Per-speaker prepared data (metadata.csv, vocab.txt, etc.)
│   ├── torgo_all_pinyin/           # Unified multi-speaker data directory
│   ├── Emilia_ZH_EN_pinyin/        # Pre-training data vocabulary reference
│   └── librispeech_pc_test_clean_cross_sentence.lst  # LibriSpeech reference list
│
├── ckpts/                          # Checkpoint directory (placeholder; not tracked)
└── ASR/                            # Downstream ASR pipeline (placeholder; not tracked)
```

**Not included in the repository** (must be prepared locally):

| Directory | Contents | How to obtain |
|-----------|----------|---------------|
| `ckpts/`  | Fine-tuned model checkpoints | Train via `scripts/run_train_RhySe-TTDS_torgo_all_sev.sh` or download from HuggingFace |
| `ASR/`    | Downstream ASR scripts, manifests, pretrained models | Scripts are provided in the paper's supplementary; pretrained models downloaded automatically |
| `exp/`    | Generated audio and evaluation CSVs | Created by running generation scripts |
| `data/TORGO/` | Raw TORGO dataset | Download from [official source](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) |

The `ckpts/` and `ASR/` directories contain `.gitkeep` placeholders so that users can populate them locally.

---

## Method Summary

### F5-TTS Baseline

The original [F5-TTS](https://github.com/SWivid/F5-TTS) serves as the baseline generator. F5-TTS provides a conditional flow matching TTS framework with a DiT (Diffusion Transformer) backbone and Sway Sampling for ODE-based inference. However, standard TTS tends to over-regularize or normalize dysarthric characteristics, making it unsuitable as-is for generating pathological speech that preserves severity-dependent acoustic patterns.

### Severity-Disentangled Chained CFG

RhySe-TTDS extends the classifier-free guidance (CFG) mechanism into a **chained, multi-branch structure** that disentangles three conditioning signals:

- **Text guidance** — ensures phonetic and textual fidelity of the generated speech.
- **Speaker guidance** — preserves speaker identity from the reference audio.
- **Severity guidance** — controls the degree of dysarthric severity (low, moderate, high).

During inference, the model computes four forward passes with progressively enriched conditioning: empty → text-only → text+speaker → text+speaker+severity. The final prediction is a chained combination of these branches, each scaled by an independent guidance strength. This allows the model to inject severity-dependent pathological patterns without collapsing into neutral or healthy-sounding speech.

The severity condition is derived from speaker-level labels in the TORGO dataset:

| Severity | Speakers |
|----------|----------|
| Low (0)  | F03, F04 |
| Moderate (1) | F01, M05 |
| High (2) | M01, M02, M04 |

### Rhythm-aware Adaptive Sway Sampling (RhySS)

RhySS introduces **utterance-level adaptive sampling** that adjusts the Sway Sampling coefficient based on the rhythm irregularity observed in the reference dysarthric speech. The rhythm irregularity score is computed from four acoustic cues:

- **Pause ratio** — proportion of silence in the utterance
- **Mean pause duration** — average length of detected pauses
- **Energy variation** — standard deviation of frame-level energy
- **Speech-rate abnormality** — deviation from a target words-per-second rate

These cues are combined into a single rhythm score that maps to an adapted Sway Sampling coefficient. Utterances with higher rhythm irregularity receive a coefficient that encourages the ODE solver to follow more of the dysarthric temporal structure, while less irregular utterances retain a coefficient closer to the standard F5-TTS setting.

### Full RhySe-TTDS

The full system combines Severity-Disentangled Chained CFG with RhySS. The severity condition is passed through the chained CFG structure, and the Sway Sampling coefficient is adapted per-utterance based on the reference speech's rhythm characteristics.

---

## Dataset

### TORGO

The primary dysarthric speech dataset used in this work is [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) (Toronto Rehab Gait and Orbit Corpus), which contains speech recordings from speakers with dysarthria and age-matched controls.

**TORGO audio and annotations are not redistributed in this repository.** Users must obtain the dataset from the official source:

> https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

The repository includes data preparation scripts (`data/prepare_torgo.py`, `data/prepare_torgo_dataset.py`) that process raw TORGO audio into the expected manifest format. Each utterance inherits the speaker-level severity label used in the experiments:

| Speaker | Severity | Severity ID |
|---------|----------|-------------|
| F01     | moderate | 1           |
| F03     | low      | 0           |
| F04     | low      | 0           |
| M01     | high     | 2           |
| M02     | high     | 2           |
| M03     | low      | 0           |
| M04     | high     | 2           |
| M05     | moderate | 1           |

### Manifest Format

The per-speaker data directories (e.g., `data/torgo_f01_pinyin/`) follow the F5-TTS manifest format:

```
wavs/<utterance_id>.wav
metadata.csv    # format: wav_name.wav|transcript_text
vocab.txt       # pinyin vocabulary (if applicable)
```

The downstream ASR evaluation uses leave-one-speaker-out (LOSO) manifests built with `ASR/build_torgo_loso_manifests.py`, with per-fold train/valid/test splits (see [Downstream ASR Augmentation](#downstream-asr-augmentation)).

---

## Installation

### Option 1: Conda Environment

```bash
conda env create -f environment.yml
conda activate f5-tts
pip install -e .
```

The environment name in `environment.yml` is `f5-tts` (retained from the upstream F5-TTS environment).

### Option 2: Pip Install

```bash
pip install -e .
```

For evaluation dependencies (Whisper, WER/CER metrics):

```bash
pip install -e ".[eval]"
```

---

## Data Preparation

1. **Download TORGO** from the [official website](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).

2. **Prepare per-speaker data** using the provided script. Adjust the speaker ID and paths as needed:

   ```bash
   # Example for speaker F01
   python data/prepare_torgo_dataset.py
   ```

   This processes raw TORGO audio into per-speaker directories with cleaned transcripts and pinyin-converted metadata.

3. **Merge into unified dataset** (for multi-speaker training):

   ```bash
   python scripts/make_torgo_all_pinyin.py
   ```

4. **Build LOSO manifests** for downstream ASR (requires the ASR pipeline scripts to be set up locally):

   ```bash
   python ASR/build_torgo_loso_manifests.py \
     --torgo_root ./data/TORGO \
     --out_dir ./ASR/manifests_torgo_loso \
     --mic_dir wav_arrayMic
   ```

5. **Verify manifest structure:**

   ```
   ASR/manifests_torgo_loso/
   ├── fold_F01/
   │   ├── train_real.csv
   │   ├── valid_real.csv
   │   └── test_real.csv
   ├── fold_F03/
   │   └── ...
   └── fold_summary.csv
   ```

---

## Checkpoints

### Pretrained F5-TTS Weights

Pretrained F5-TTS-compatible weights are required and will be automatically downloaded from HuggingFace on first use:

> https://huggingface.co/SWivid/F5-TTS

### RhySe-TTDS Fine-tuned Checkpoints

Fine-tuned checkpoints should be placed under `ckpts/` (the directory is a placeholder in the repository):

- **Unified severity-aware checkpoint:** `ckpts/torgo_all_sev_4gpu/model_last.pt` — used by the Chained CFG and RhySS experiments.
- **Per-speaker checkpoints:** `ckpts/torgo_f01/model_last.pt`, `ckpts/torgo_f03/model_last.pt`, etc. — used by the F5-TTS baseline per-speaker experiments.

Checkpoints are **not included in the repository**. Users must train or obtain them separately. Multi-GPU training is supported via the provided training script:

```bash
bash scripts/run_train_RhySe-TTDS_torgo_all_sev.sh
```

---

## Running Speech Generation Experiments

The repository provides four generation configurations corresponding to the experimental settings in the paper. Each configuration has a shell script that orchestrates batch generation and evaluation across all TORGO speakers.

### 8.1 F5-TTS Baseline

The original F5-TTS generation with per-speaker fine-tuned checkpoints and fixed Sway Sampling.

```bash
bash scripts/run_batch_f5tts_eval2.sh
```

**Output directory:** `exp/f5tts_torgo_baseline_generation_eval_no_wer_cer/`

### 8.2 RhySe-TTDS with Severity Adaptive Sway Sampling

Generation with severity-aware adaptive Sway Sampling (without Chained CFG).

```bash
bash scripts/run_batch_RhySe-TTDS_eval2_adaptive_ss.sh
```

**Output directory:** `exp/f5ttds_torgo_severity_adaptive_ss_eval/`

### 8.3 RhySe-TTDS with Chained CFG (Fixed Sway)

Generation with Severity-Disentangled Chained CFG and fixed Sway Sampling.

```bash
bash scripts/run_batch_RhySe-TTDS_chained_cfg_eval.sh
```

**Output directory:** `exp/f5ttds_chained_cfg_fixedsway_eval/`

### 8.4 Full RhySe-TTDS (Chained CFG + RhySS)

Generation combining Severity-Disentangled Chained CFG with Rhythm-aware Adaptive Sway Sampling (RhySS).

```bash
bash scripts/run_batch_RhySe-TTDS_eval2_rhythm_adaptive_ss.sh
```

**Output directory:** `exp/f5ttds_chained_cfg_rhyss_eval/`

> **Note:** Each script requires modifying the `PROJECT_ROOT` and `DATA_ROOT` paths at the top of the file to match your local setup. Set `MAX_ITEMS=8` for a quick debug run, then set `MAX_ITEMS=-1` for full generation.

---

## Output Files and Evaluation Metrics

Generation and evaluation scripts create the following outputs locally under each experiment directory (not tracked by Git):

```
exp/<experiment_name>/
├── generated_wavs/<speaker>/<utterance_id>.wav   # Generated audio
├── auto_manifest.csv                              # Generated utterance manifest
├── generation_log.csv                             # Per-utterance generation log
├── per_utterance_metrics.csv                      # Per-utterance objective metrics
├── summary_overall.csv                            # Aggregate metrics across all speakers
├── summary_by_speaker.csv                         # Metrics broken down by speaker
├── summary_by_severity.csv                        # Metrics broken down by severity level
├── summary_by_speaker_severity.csv                # Metrics by speaker × severity
└── failed_items.csv                               # Failed generation items (if any)
```

### Evaluation Metrics

The following objective metrics are computed to assess whether generated speech preserves dysarthria-related rhythm and acoustic characteristics:

| Metric | Description |
|--------|-------------|
| **SIM-o** | Speaker similarity ( cosine similarity of speaker embeddings) |
| **AutoPCP** | Automatic pronunciation quality proxy |
| **F0 correlation** | Pearson correlation of F0 contour between reference and generated speech |
| **Energy correlation** | Pearson correlation of energy contour |
| **Pause correlation** | Correlation of pause patterns |
| **Generated pause ratio** | Proportion of silence in generated speech |
| **Reference pause ratio** | Proportion of silence in reference speech |
| **Abs. pause-ratio diff.** | Absolute difference between generated and reference pause ratios |
| **Speech rate** | Words (or phonemes) per second in generated speech |
| **Generation time** | Wall-clock time for inference |
| **RTF** | Real-time factor (generation time / audio duration) |

---

## Downstream ASR Augmentation

The downstream ASR pipeline evaluates whether synthetic dysarthric speech improves recognition performance. The ASR scripts, manifests, and pretrained models are **not included in this repository** and must be set up locally. The expected directory layout is:

```
ASR/
├── Whisper/                        # Whisper fine-tuning and evaluation
├── Hubert/                         # HuBERT fine-tuning and evaluation
├── wav2vec/                        # wav2vec2 fine-tuning and evaluation
├── build_torgo_loso_manifests.py   # Build LOSO train/valid/test manifests
├── make_real_f5ttds_manifests.py   # Build real-plus-synthetic manifests
├── manifests_torgo_loso/           # LOSO manifests (per-fold train/valid/test)
└── manifests_torgo_real_plus_f5ttds/  # Real + RhySe-TTDS synthetic manifests
```

Three ASR architectures are evaluated:

### ASR Evaluation Workflow

1. **Prepare LOSO manifests:** Real-only TORGO manifests are built using `ASR/build_torgo_loso_manifests.py`, producing per-fold train/valid/test splits.

2. **Generate synthetic speech:** Use the generation scripts above to produce synthetic dysarthric audio.

3. **Build real-plus-synthetic manifests:** Merge real and synthetic data into augmented training manifests:

   ```bash
   python ASR/make_real_f5ttds_manifests.py \
     --real_root ./ASR/manifests_torgo_loso \
     --synth_csv ./ASR/synth_f5ttds_all.csv \
     --out_root ./ASR/manifests_torgo_real_plus_f5ttds \
     --min_duration 0.5 \
     --synth_ratio 1.0
   ```

4. **Fine-tune ASR models:** Run per-fold ASR training for each setting (real-only, real+F5-TTS-baseline, real+RhySe-TTDS):

   ```bash
   # HuBERT: real-only baseline
   bash ASR/Hubert/run_train.sh

   # HuBERT: real + RhySe-TTDS augmentation
   bash ASR/Hubert/run_train_real+f5-ttds.sh

   # Whisper: real + RhySe-TTDS augmentation
   bash ASR/Whisper/run_train_real+f5-ttds.sh

   # wav2vec2: real + RhySe-TTDS augmentation
   bash ASR/wav2vec/run_train_real+f5-ttds.sh
   ```

5. **Collect results:** Each ASR pipeline includes a `summarize_*_results.py` script that aggregates per-fold WER/CER into speaker-level and severity-level summary CSVs.

> **Note:** The ASR scripts, manifests, and pretrained models are not included in this repository. They must be set up locally following the directory structure described above.

### ASR Evaluation Settings

Each ASR model is evaluated under three training data conditions:

| Setting | Training Data |
|---------|---------------|
| `real_only` | Real TORGO speech only |
| `real_plus_f5tts` | Real + F5-TTS baseline synthetic speech |
| `f5ttds` | Real + RhySe-TTDS synthetic speech |

Evaluation is performed in a LOSO protocol: for each test speaker, the ASR model is trained on all other speakers' data.

---

## Reproducing the Paper Experiments

1. **Install the environment** (see [Installation](#installation)).

2. **Prepare TORGO data** (see [Data Preparation](#data-preparation)).

3. **Obtain or train checkpoints:**
   - Download pretrained F5-TTS weights (auto-downloaded from HuggingFace).
   - Train the unified severity-aware checkpoint:
     ```bash
     bash scripts/run_train_RhySe-TTDS_torgo_all_sev.sh
     ```

4. **Run generation experiments:**
   ```bash
   # F5-TTS baseline
   bash scripts/run_batch_f5tts_eval2.sh

   # RhySe-TTDS with severity adaptive SS
   bash scripts/run_batch_RhySe-TTDS_eval2_adaptive_ss.sh

   # RhySe-TTDS with Chained CFG (fixed sway)
   bash scripts/run_batch_RhySe-TTDS_chained_cfg_eval.sh

   # Full RhySe-TTDS (Chained CFG + RhySS)
   bash scripts/run_batch_RhySe-TTDS_eval2_rhythm_adaptive_ss.sh
   ```

5. **Build ASR manifests** (requires ASR pipeline scripts set up locally):
   ```bash
   python ASR/make_real_f5ttds_manifests.py \
     --real_root ./ASR/manifests_torgo_loso \
     --synth_csv ./ASR/synth_f5ttds_all.csv \
     --out_root ./ASR/manifests_torgo_real_plus_f5ttds
   ```

6. **Run ASR evaluation** (requires ASR pipeline set up locally):
   ```bash
   bash ASR/Hubert/run_train_real+f5-ttds.sh
   bash ASR/Whisper/run_train_real+f5-ttds.sh
   bash ASR/wav2vec/run_train_real+f5-ttds.sh
   ```

7. **Collect summary CSVs** from each experiment and ASR output directory.

---

## Important Notes

- **TORGO data are not redistributed.** Users must obtain the dataset from the [official source](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) and prepare it using the provided scripts.
- **Large files are excluded from the repository.** The `ckpts/` and `ASR/` directories contain only `.gitkeep` placeholders. Fine-tuned checkpoints must be trained or downloaded separately. The ASR pipeline scripts and pretrained models must be set up locally.


---

## Citation

```bibtex
@article{rhyse_ttds,
  title   = {RhySe-TTDS: Rhythm-Aware and Severity-Disentangled Dysarthric Speech Synthesis for ASR Augmentation},
  author  = {Anonymous},
  journal = {To appear},
  year    = {2026}
}
```

If you use F5-TTS in your work, please also cite:

```bibtex
@article{chen2024f5tts,
  title   = {F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author  = {Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kaicheng Yu and Xie Chen},
  journal = {arXiv preprint arXiv:2410.06885},
  year    = {2024}
}
```

---

## Acknowledgements

RhySe-TTDS is built upon [F5-TTS](https://github.com/SWivid/F5-TTS) and uses the [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) corpus for dysarthric speech research. We thank the authors of both projects for making their work publicly available.

---

## License

The license will be updated upon publication.
