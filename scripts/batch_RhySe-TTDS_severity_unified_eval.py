#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference/evaluation for unified severity-aware F5-TTDS checkpoint.

This script is separated from the old batch_f5tts_eval2_adaptive_ss.py
so that the original baseline/adaptive-sway script can be preserved for ablation.

Main differences:
1. Use one unified checkpoint for all TORGO speakers:
   ckpts/torgo_all_sev_4gpu/model_last.pt

2. Pass severity condition into infer_process -> model.sample -> DiT.

3. Keep adaptive_sway_mode as an independent switch:
   fixed    : fixed F5-TTS Sway coefficient
   severity : severity-aware adaptive Sway coefficient

Ablation examples:
- Severity embedding only:
  --severity_condition_mode speaker --adaptive_sway_mode fixed

- No severity condition:
  --severity_condition_mode none --adaptive_sway_mode fixed

- Severity embedding + adaptive sway:
  --severity_condition_mode speaker --adaptive_sway_mode severity
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Reuse utilities from your old script.
from batch_f5tts_eval2_rhythm_adaptive_ss import (
    DirectF5TTSGenerator,
    SpeakerSimilarity,
    build_manifest_from_data_root,
    evaluate_row,
    numeric_mean_std,
    resolve_sway_coef_for_row,
    safe_str,
    sanitize_filename,
    setup_logger,
    write_metric_readme,
    write_rating_sheet,
)


SEVERITY_TO_ID = {
    "low": 0,
    "mild": 0,
    "moderate": 1,
    "mid": 1,
    "medium": 1,
    "high": 2,
    "severe": 2,
}


def severity_to_id(severity: str) -> Optional[int]:
    severity = safe_str(severity).strip().lower()
    if severity in SEVERITY_TO_ID:
        return int(SEVERITY_TO_ID[severity])
    return None


class UnifiedSeverityF5TTSGenerator(DirectF5TTSGenerator):
    """
    Override only generate_one() so that severity can be passed into infer_process().
    Everything else, including model loading and vocoder loading, is reused.
    """

    def generate_one(
        self,
        ref_audio: Path,
        ref_text: str,
        gen_text: str,
        ckpt_file: Path,
        vocab_file: Optional[Path],
        output_wav: Path,
        sway_sampling_coef: Optional[float] = None,
        severity_id: Optional[int] = None,
    ) -> float:
        if not ref_audio.exists():
            raise FileNotFoundError(f"ref_audio not found: {ref_audio}")

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        model_obj = self.get_model(ckpt_file, vocab_file)

        ref_audio_processed, ref_text_processed = self.preprocess_ref_audio_text(str(ref_audio), ref_text)

        local_sway_sampling_coef = (
            float(self.args.sway_sampling_coef)
            if sway_sampling_coef is None
            else float(sway_sampling_coef)
        )

        start = time.perf_counter()

        audio_segment, final_sample_rate, _spectrogram = self.infer_process(
            ref_audio_processed,
            ref_text_processed,
            gen_text,
            model_obj,
            self.vocoder,
            mel_spec_type=self.vocoder_name,
            target_rms=self.args.target_rms,
            cross_fade_duration=self.args.cross_fade_duration,
            nfe_step=self.args.nfe_step,
            cfg_strength=self.args.cfg_strength,
            sway_sampling_coef=local_sway_sampling_coef,
            severity=severity_id,
            speed=self.args.speed,
            fix_duration=self.args.fix_duration if self.args.fix_duration > 0 else None,
            device=self.args.device,
        )

        elapsed = time.perf_counter() - start
        self.sf.write(str(output_wav), audio_segment, final_sample_rate)

        if bool(self.args.remove_silence):
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
            remove_silence_for_generated_wav(str(output_wav))

        return float(elapsed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified severity-aware F5-TTDS batch inference/evaluation."
    )

    # Paths
    p.add_argument("--project_root", default="/home/user/ly/F5-TTS-main")
    p.add_argument("--data_root", default="/home/user/ly/F5-TTS-main/data")
    p.add_argument("--speaker_glob", default="torgo_*_pinyin")
    p.add_argument("--metadata_name", default="metadata.csv")
    p.add_argument("--audio_subdir", default="wavs")
    p.add_argument("--out_dir", default="/home/user/ly/F5-TTS-main/exp/f5ttds_unified_sevcond_eval")

    # Unified checkpoint
    p.add_argument(
        "--ckpt_file",
        default="/home/user/ly/F5-TTS-main/ckpts/torgo_all_sev_4gpu/model_last.pt",
        help="Unified severity-aware F5-TTDS checkpoint.",
    )
    p.add_argument(
        "--vocab_file",
        default="/home/user/ly/F5-TTS-main/data/torgo_all_pinyin/vocab.txt",
        help="Vocab used during unified severity-aware training.",
    )

    # Model settings
    p.add_argument("--model_name", default="F5TTS_v1_Base")
    p.add_argument("--model_cfg", default="")
    p.add_argument("--vocoder_name", default="vocos", choices=["vocos", "bigvgan"])
    p.add_argument("--load_vocoder_from_local", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_ema", type=int, default=0)

    # Inference settings
    p.add_argument("--nfe_step", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    p.add_argument(
        "--adaptive_sway_mode",
        default="fixed",
        choices=["fixed", "severity"],
        help="fixed: use one Sway coefficient; severity: choose Sway coefficient by severity.",
    )
    p.add_argument(
        "--severity_condition_mode",
        default="speaker",
        choices=["none", "speaker"],
        help="none: do not pass severity to the model; speaker: infer severity from speaker label.",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--cross_fade_duration", type=float, default=0.15)
    p.add_argument("--fix_duration", type=float, default=-1.0)
    p.add_argument("--remove_silence", type=int, default=0)

    # Reference selection
    p.add_argument("--ref_strategy", default="longest_text", choices=["first", "longest_text", "by_index", "random", "self"])
    p.add_argument("--ref_index", type=int, default=0)
    p.add_argument("--exclude_ref_from_generation", type=int, default=1)

    # Run control
    p.add_argument("--skip_existing", type=int, default=1)
    p.add_argument("--overwrite_existing", type=int, default=0)
    p.add_argument("--eval_only", type=int, default=0)
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--max_items", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_auto_manifest", type=int, default=1)

    # Evaluation settings
    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--enable_sim", type=int, default=1)
    p.add_argument("--sim_backend", default="speechbrain", choices=["speechbrain", "none"])
    p.add_argument("--sim_model", default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--pause_top_db", type=float, default=35.0)
    p.add_argument("--frame_ms", type=float, default=25.0)
    p.add_argument("--hop_ms", type=float, default=10.0)
    p.add_argument("--min_pause_ms", type=float, default=120.0)
    p.add_argument("--f0_min", type=float, default=50.0)
    p.add_argument("--f0_max", type=float, default=500.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    gen_dir = out_dir / "generated_wavs"

    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(out_dir / "run.log")

    ckpt_file = Path(args.ckpt_file).expanduser().resolve()
    vocab_file = Path(args.vocab_file).expanduser().resolve() if safe_str(args.vocab_file).strip() else None

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Unified checkpoint not found: {ckpt_file}")

    if vocab_file is not None and not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

    logging.info("Unified severity-aware F5-TTDS inference")
    logging.info("project_root = %s", project_root)
    logging.info("ckpt_file    = %s", ckpt_file)
    logging.info("vocab_file   = %s", vocab_file)
    logging.info("out_dir      = %s", out_dir)
    logging.info("severity_condition_mode = %s", args.severity_condition_mode)
    logging.info("adaptive_sway_mode      = %s", args.adaptive_sway_mode)

    df = build_manifest_from_data_root(
        data_root=args.data_root,
        speaker_glob=args.speaker_glob,
        metadata_name=args.metadata_name,
        audio_subdir=args.audio_subdir,
        ref_strategy=args.ref_strategy,
        ref_index=args.ref_index,
        random_seed=args.seed,
        exclude_ref_from_generation=bool(args.exclude_ref_from_generation),
    )

    if args.max_items is not None and args.max_items > 0:
        df = df.head(args.max_items).copy()
        logging.info("Debug mode: only first %d items will be used.", args.max_items)

    if args.severity_condition_mode == "speaker":
        df["severity_id_used"] = df["severity"].map(lambda x: severity_to_id(x))
    else:
        df["severity_id_used"] = None

    df["ckpt_file"] = str(ckpt_file)
    df["vocab_file"] = str(vocab_file) if vocab_file is not None else ""
    df["severity_condition_mode"] = args.severity_condition_mode

    if args.save_auto_manifest:
        df.to_csv(out_dir / "auto_manifest.csv", index=False, encoding="utf-8-sig")
        logging.info("Saved auto manifest: %s", out_dir / "auto_manifest.csv")
        logging.info("Manifest preview:\n%s", df.head(5).to_string(index=False))

    config = vars(args).copy()
    config["resolved_ckpt_file"] = str(ckpt_file)
    config["resolved_vocab_file"] = str(vocab_file) if vocab_file is not None else ""
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metric_readme(out_dir / "README_metrics.md")

    if args.dry_run:
        preview_rows = []
        for _, row_s in df.iterrows():
            row = row_s.to_dict()
            speaker = sanitize_filename(row.get("speaker", "unknown"))
            utt_id = sanitize_filename(row.get("utt_id", "utt"))
            output_wav = gen_dir / speaker / f"{utt_id}.wav"
            preview_rows.append({
                **row,
                "output_wav": str(output_wav),
                "sway_sampling_coef_used": resolve_sway_coef_for_row(row, args),
                "dry_status": "ok",
            })
        pd.DataFrame(preview_rows).to_csv(out_dir / "dry_run_preview.csv", index=False, encoding="utf-8-sig")
        logging.info("Dry run finished. See %s", out_dir / "dry_run_preview.csv")
        return

    generation_records = []
    failed_records = []

    if not bool(args.eval_only):
        generator = UnifiedSeverityF5TTSGenerator(args)

        for _, row_s in df.iterrows():
            row = row_s.to_dict()
            speaker = sanitize_filename(row.get("speaker", "unknown_speaker"))
            utt_id = sanitize_filename(row.get("utt_id", "utt"))
            output_wav = gen_dir / speaker / f"{utt_id}.wav"

            local_sway_coef = resolve_sway_coef_for_row(row, args)
            severity_id = row.get("severity_id_used", None)

            if pd.isna(severity_id):
                severity_id = None
            elif severity_id is not None:
                severity_id = int(severity_id)

            rec: Dict[str, Any] = {
                "utt_id": row.get("utt_id", utt_id),
                "speaker": row.get("speaker", ""),
                "severity": row.get("severity", ""),
                "severity_id_used": severity_id if severity_id is not None else "",
                "severity_condition_mode": args.severity_condition_mode,
                "adaptive_sway_mode": args.adaptive_sway_mode,
                "sway_sampling_coef_used": float(local_sway_coef),
                "gen_audio": str(output_wav),
                "ckpt_file": str(ckpt_file),
                "vocab_file": str(vocab_file) if vocab_file is not None else "",
                "use_ema": int(args.use_ema),
            }

            try:
                if bool(args.skip_existing) and not bool(args.overwrite_existing) and output_wav.exists() and output_wav.stat().st_size > 0:
                    rec.update({
                        "generation_status": "skipped_existing",
                        "generation_time_sec": 0.0,
                        "generation_returncode": 0,
                    })
                    logging.info(
                        "Skip existing: %s | speaker=%s | severity=%s | severity_id=%s | sway=%s",
                        output_wav,
                        speaker,
                        row.get("severity", ""),
                        severity_id,
                        local_sway_coef,
                    )
                else:
                    ref_audio = Path(safe_str(row["ref_audio"])).expanduser().resolve()
                    ref_text = safe_str(row["ref_text"]).strip()
                    gen_text = safe_str(row["gen_text"]).strip()

                    if not ref_audio.exists():
                        raise FileNotFoundError(f"ref_audio not found: {ref_audio}")
                    if not ref_text or not gen_text:
                        raise ValueError("ref_text/gen_text is empty.")

                    logging.info(
                        "Generating %s | speaker=%s | severity=%s | severity_id=%s | sway=%s | mode=%s",
                        utt_id,
                        speaker,
                        row.get("severity", ""),
                        severity_id,
                        local_sway_coef,
                        args.adaptive_sway_mode,
                    )

                    elapsed = generator.generate_one(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        gen_text=gen_text,
                        ckpt_file=ckpt_file,
                        vocab_file=vocab_file,
                        output_wav=output_wav,
                        sway_sampling_coef=local_sway_coef,
                        severity_id=severity_id,
                    )

                    rec.update({
                        "generation_status": "ok",
                        "generation_time_sec": float(elapsed),
                        "generation_returncode": 0,
                    })

            except Exception as e:
                logging.exception("Generation failed: utt_id=%s", row.get("utt_id", ""))
                rec.update({
                    "generation_status": "exception",
                    "generation_error": str(e),
                    "generation_returncode": 1,
                    "generation_time_sec": np.nan,
                })
                failed_records.append({**row, **rec})

            generation_records.append(rec)

        gen_df = pd.DataFrame(generation_records)
        gen_df.to_csv(out_dir / "generation_log.csv", index=False, encoding="utf-8-sig")

        merge_cols = [
            "utt_id",
            "gen_audio",
            "generation_status",
            "generation_time_sec",
            "adaptive_sway_mode",
            "sway_sampling_coef_used",
            "severity_id_used",
            "severity_condition_mode",
        ]
        merge_cols = [c for c in merge_cols if c in gen_df.columns]
        df = df.merge(gen_df[merge_cols], on="utt_id", how="left")
    else:
        df["generation_status"] = "eval_only"
        if "generation_time_sec" not in df.columns:
            df["generation_time_sec"] = np.nan

    sim_model = SpeakerSimilarity(
        enable=bool(args.enable_sim),
        backend=args.sim_backend,
        model_name=args.sim_model,
        device=args.device,
    )

    eval_records = []

    for _, row_s in df.iterrows():
        row = row_s.to_dict()
        gen_audio_s = safe_str(row.get("gen_audio", "")).strip()

        if not gen_audio_s:
            failed_records.append({**row, "eval_status": "missing_gen_audio"})
            continue

        gen_audio = Path(gen_audio_s).expanduser().resolve()

        if not gen_audio.exists():
            failed_records.append({**row, "eval_status": "gen_audio_not_found"})
            continue

        try:
            logging.info("Evaluating %s", row.get("utt_id", gen_audio.name))
            eval_records.append(evaluate_row(row, gen_audio, args, sim_model))
        except Exception as e:
            logging.exception("Evaluation failed: utt_id=%s", row.get("utt_id", ""))
            failed_records.append({**row, "eval_status": "exception", "eval_error": str(e)})

    eval_df = pd.DataFrame(eval_records)

    if len(eval_df):
        eval_df.to_csv(out_dir / "per_utterance_metrics.csv", index=False, encoding="utf-8-sig")
        numeric_mean_std(eval_df, None, out_dir / "summary_overall.csv")
        numeric_mean_std(eval_df, ["speaker"], out_dir / "summary_by_speaker.csv")
        if "severity" in eval_df.columns:
            numeric_mean_std(eval_df, ["severity"], out_dir / "summary_by_severity.csv")
            numeric_mean_std(eval_df, ["speaker", "severity"], out_dir / "summary_by_speaker_severity.csv")
        write_rating_sheet(eval_df, out_dir / "mos_smos_rating_sheet.csv")
        logging.info("Saved metrics to %s", out_dir / "per_utterance_metrics.csv")
    else:
        logging.warning("No successful evaluation records.")

    if failed_records:
        pd.DataFrame(failed_records).to_csv(out_dir / "failed_items.csv", index=False, encoding="utf-8-sig")
        logging.warning("Saved failed items to %s", out_dir / "failed_items.csv")

    logging.info("Done. Output directory: %s", out_dir)

    gc.collect()


if __name__ == "__main__":
    main()