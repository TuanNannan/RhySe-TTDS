#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full F5-TTDS batch inference/evaluation wrapper.

This wrapper combines:
  1) Chained CFG from batch_f5ttds_chained_cfg_eval.py
  2) Utterance-level pathological rhythm-aware Adaptive Sway Sampling (RhySS)

It intentionally keeps the trained F5-TTDS model architecture and checkpoint unchanged.
RhySS only changes the per-utterance sway_sampling_coef passed into infer_process().

Expected chain:
    batch_f5ttds_chained_cfg_rhyss_eval.py
        -> batch_f5ttds_severity_unified_eval.main()
        -> FullF5TTDSGenerator.generate_one()
        -> infer_process(..., chained_cfg=True, sway_sampling_coef=adaptive_sway, ...)
        -> CFM.sample(..., chained_cfg=True, sway_sampling_coef=adaptive_sway, ...)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as AF

import batch_f5ttds_severity_unified_eval as base


_ORIGINAL_PARSE_ARGS = base.parse_args


def _str_to_bool01(x) -> bool:
    return bool(int(x)) if isinstance(x, str) and x.strip() in {"0", "1"} else bool(x)


def parse_severity_id_to_label(spec: str) -> Dict[int, str]:
    """Parse a mapping string such as '0:low,1:moderate,2:high'."""
    out: Dict[int, str] = {}
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        try:
            out[int(k.strip())] = v.strip().lower()
        except ValueError:
            continue
    return out or {0: "low", 1: "moderate", 2: "high"}


def clip_float(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def parse_args_full():
    """
    Parse full F5-TTDS-specific arguments first, then pass all remaining args
    to the original severity-unified evaluation parser.
    """
    full_parser = argparse.ArgumentParser(add_help=False)

    # Chained CFG arguments
    full_parser.add_argument("--chained_cfg", type=int, default=1)
    full_parser.add_argument("--text_cfg_strength", type=float, default=1.5)
    full_parser.add_argument("--speaker_cfg_strength", type=float, default=1.0)
    full_parser.add_argument("--severity_cfg_strength", type=float, default=0.5)

    # RhySS arguments
    full_parser.add_argument(
        "--adaptive_sway_mode",
        default="rhythm",
        choices=["fixed", "severity", "rhythm"],
        help=(
            "fixed: use args.sway_sampling_coef; "
            "severity: use severity-dependent base coefficient; "
            "rhythm: use severity-dependent base coefficient plus rhythm-score correction."
        ),
    )
    full_parser.add_argument(
        "--severity_id_to_label",
        default="0:low,1:moderate,2:high",
        help="Mapping from severity integer id to label, e.g., '0:low,1:moderate,2:high'.",
    )
    full_parser.add_argument("--low_sway_coef", type=float, default=-0.70)
    full_parser.add_argument("--moderate_sway_coef", type=float, default=-0.85)
    full_parser.add_argument("--high_sway_coef", type=float, default=-0.95)
    full_parser.add_argument("--sway_min_coef", type=float, default=-1.0)
    full_parser.add_argument("--sway_max_coef", type=float, default=0.0)
    full_parser.add_argument("--rhythm_sway_gamma", type=float, default=0.20)
    full_parser.add_argument("--rhythm_target_wps", type=float, default=2.5)
    full_parser.add_argument("--rhythm_max_energy_delta_db", type=float, default=10.0)
    full_parser.add_argument("--rhythm_max_mean_pause_sec", type=float, default=1.0)
    full_parser.add_argument("--rhythm_weight_pause", type=float, default=0.40)
    full_parser.add_argument("--rhythm_weight_mean_pause", type=float, default=0.25)
    full_parser.add_argument("--rhythm_weight_energy", type=float, default=0.20)
    full_parser.add_argument("--rhythm_weight_rate", type=float, default=0.15)
    full_parser.add_argument("--rhythm_frame_ms", type=float, default=40.0)
    full_parser.add_argument("--rhythm_hop_ms", type=float, default=10.0)
    full_parser.add_argument("--rhythm_pause_top_db", type=float, default=35.0)
    full_parser.add_argument("--rhythm_min_pause_ms", type=float, default=200.0)
    full_parser.add_argument("--rhyss_aux_log_name", default="rhyss_aux_generation_log.csv")

    full_args, remaining_argv = full_parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining_argv
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = old_argv

    # Chained CFG
    args.chained_cfg = bool(full_args.chained_cfg)
    args.text_cfg_strength = float(full_args.text_cfg_strength)
    args.speaker_cfg_strength = float(full_args.speaker_cfg_strength)
    args.severity_cfg_strength = float(full_args.severity_cfg_strength)

    # RhySS
    args.adaptive_sway_mode = str(full_args.adaptive_sway_mode)
    args.severity_id_to_label = str(full_args.severity_id_to_label)
    args.low_sway_coef = float(full_args.low_sway_coef)
    args.moderate_sway_coef = float(full_args.moderate_sway_coef)
    args.high_sway_coef = float(full_args.high_sway_coef)
    args.sway_min_coef = float(full_args.sway_min_coef)
    args.sway_max_coef = float(full_args.sway_max_coef)
    args.rhythm_sway_gamma = float(full_args.rhythm_sway_gamma)
    args.rhythm_target_wps = float(full_args.rhythm_target_wps)
    args.rhythm_max_energy_delta_db = float(full_args.rhythm_max_energy_delta_db)
    args.rhythm_max_mean_pause_sec = float(full_args.rhythm_max_mean_pause_sec)
    args.rhythm_weight_pause = float(full_args.rhythm_weight_pause)
    args.rhythm_weight_mean_pause = float(full_args.rhythm_weight_mean_pause)
    args.rhythm_weight_energy = float(full_args.rhythm_weight_energy)
    args.rhythm_weight_rate = float(full_args.rhythm_weight_rate)
    args.rhythm_frame_ms = float(full_args.rhythm_frame_ms)
    args.rhythm_hop_ms = float(full_args.rhythm_hop_ms)
    args.rhythm_pause_top_db = float(full_args.rhythm_pause_top_db)
    args.rhythm_min_pause_ms = float(full_args.rhythm_min_pause_ms)
    args.rhyss_aux_log_name = str(full_args.rhyss_aux_log_name)

    return args


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", str(text)))


def load_mono_audio(path: Path, target_sr: int = 24000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def estimate_reference_rhythm_score(ref_audio: Path, ref_text: str, args: argparse.Namespace) -> Dict[str, float]:
    """
    Estimate an utterance-level pathological rhythm irregularity score from the
    reference speech. This is not frame-level ODE step allocation; it adapts the
    global Sway coefficient of the current utterance.
    """
    try:
        wav, sr = load_mono_audio(ref_audio, target_sr=24000)
        if wav.numel() < 16:
            raise ValueError("empty or too short audio")

        frame_len = max(16, int(sr * float(args.rhythm_frame_ms) / 1000.0))
        hop_len = max(1, int(sr * float(args.rhythm_hop_ms) / 1000.0))
        if wav.numel() < frame_len:
            frame_len = wav.numel()
            hop_len = max(1, frame_len // 2)

        frames = wav.unfold(0, frame_len, hop_len)
        rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-10)
        db = 20.0 * torch.log10(rms + 1e-10)
        threshold = torch.max(db) - float(args.rhythm_pause_top_db)
        silent = db < threshold

        pause_ratio = float(silent.float().mean().item()) if silent.numel() > 0 else 0.0

        min_pause_frames = max(1, int(float(args.rhythm_min_pause_ms) / float(args.rhythm_hop_ms)))
        silent_list = silent.cpu().tolist()
        pause_durs = []
        cur = 0
        for v in silent_list:
            if v:
                cur += 1
            else:
                if cur >= min_pause_frames:
                    pause_durs.append(cur * hop_len / sr)
                cur = 0
        if cur >= min_pause_frames:
            pause_durs.append(cur * hop_len / sr)
        mean_pause_sec = float(sum(pause_durs) / len(pause_durs)) if pause_durs else 0.0
        mean_pause_norm = clip_float(mean_pause_sec / max(float(args.rhythm_max_mean_pause_sec), 1e-8), 0.0, 1.0)

        if db.numel() > 1:
            energy_delta_db = float(torch.mean(torch.abs(db[1:] - db[:-1])).item())
        else:
            energy_delta_db = 0.0
        energy_var_norm = clip_float(energy_delta_db / max(float(args.rhythm_max_energy_delta_db), 1e-8), 0.0, 1.0)

        duration_sec = float(wav.numel() / sr)
        speech_duration_sec = max(duration_sec * (1.0 - pause_ratio), 1e-8)
        n_words = count_words(ref_text)
        if n_words > 0:
            speech_rate_wps = float(n_words / speech_duration_sec)
            rate_abn = clip_float(
                abs(speech_rate_wps - float(args.rhythm_target_wps)) / max(float(args.rhythm_target_wps), 1e-8),
                0.0,
                1.0,
            )
        else:
            speech_rate_wps = math.nan
            rate_abn = 0.0

        score = (
            float(args.rhythm_weight_pause) * pause_ratio
            + float(args.rhythm_weight_mean_pause) * mean_pause_norm
            + float(args.rhythm_weight_energy) * energy_var_norm
            + float(args.rhythm_weight_rate) * rate_abn
        )
        score = clip_float(score, 0.0, 1.0)
        return {
            "rhythm_score": score,
            "rhythm_pause_ratio": pause_ratio,
            "rhythm_mean_pause_sec": mean_pause_sec,
            "rhythm_mean_pause_norm": mean_pause_norm,
            "rhythm_energy_delta_db": energy_delta_db,
            "rhythm_energy_var_norm": energy_var_norm,
            "rhythm_speech_rate_wps": speech_rate_wps,
            "rhythm_rate_abn": rate_abn,
        }
    except Exception as e:
        print(f"[RhySS][WARN] Failed to estimate rhythm score for {ref_audio}: {e}")
        return {
            "rhythm_score": math.nan,
            "rhythm_pause_ratio": math.nan,
            "rhythm_mean_pause_sec": math.nan,
            "rhythm_mean_pause_norm": math.nan,
            "rhythm_energy_delta_db": math.nan,
            "rhythm_energy_var_norm": math.nan,
            "rhythm_speech_rate_wps": math.nan,
            "rhythm_rate_abn": math.nan,
        }


def severity_label_from_id(severity_id: Optional[int], args: argparse.Namespace) -> str:
    mapping = parse_severity_id_to_label(getattr(args, "severity_id_to_label", "0:low,1:moderate,2:high"))
    if severity_id is None:
        return "unknown"
    try:
        return mapping.get(int(severity_id), "unknown")
    except Exception:
        return "unknown"


def base_sway_from_severity(severity_label: str, args: argparse.Namespace, default_coef: float) -> float:
    sev = str(severity_label).strip().lower()
    if sev in {"low", "mild"}:
        coef = float(args.low_sway_coef)
    elif sev in {"moderate", "mid", "medium"}:
        coef = float(args.moderate_sway_coef)
    elif sev in {"high", "severe"}:
        coef = float(args.high_sway_coef)
    else:
        coef = float(default_coef)
    return clip_float(coef, float(args.sway_min_coef), float(args.sway_max_coef))


def resolve_rhyss_sway(
    ref_audio: Path,
    ref_text: str,
    severity_id: Optional[int],
    default_sway: float,
    args: argparse.Namespace,
) -> Dict[str, float | str | int | None]:
    mode = str(getattr(args, "adaptive_sway_mode", "fixed")).strip().lower()
    sev_label = severity_label_from_id(severity_id, args)

    if mode == "fixed":
        coef = float(default_sway)
        rhythm_info = {k: math.nan for k in [
            "rhythm_score", "rhythm_pause_ratio", "rhythm_mean_pause_sec", "rhythm_mean_pause_norm",
            "rhythm_energy_delta_db", "rhythm_energy_var_norm", "rhythm_speech_rate_wps", "rhythm_rate_abn",
        ]}
        return {
            "adaptive_sway_mode": mode,
            "severity_id": severity_id,
            "severity_label": sev_label,
            "base_sway_coef": coef,
            "sway_sampling_coef_used": coef,
            **rhythm_info,
        }

    base_coef = base_sway_from_severity(sev_label, args, default_coef=default_sway)

    if mode == "severity":
        rhythm_info = {k: math.nan for k in [
            "rhythm_score", "rhythm_pause_ratio", "rhythm_mean_pause_sec", "rhythm_mean_pause_norm",
            "rhythm_energy_delta_db", "rhythm_energy_var_norm", "rhythm_speech_rate_wps", "rhythm_rate_abn",
        ]}
        return {
            "adaptive_sway_mode": mode,
            "severity_id": severity_id,
            "severity_label": sev_label,
            "base_sway_coef": base_coef,
            "sway_sampling_coef_used": base_coef,
            **rhythm_info,
        }

    if mode == "rhythm":
        rhythm_info = estimate_reference_rhythm_score(ref_audio, ref_text, args)
        rhythm_score = rhythm_info.get("rhythm_score", math.nan)
        score_for_coef = 0.0 if not math.isfinite(float(rhythm_score)) else float(rhythm_score)
        coef = base_coef - float(args.rhythm_sway_gamma) * score_for_coef
        coef = clip_float(coef, float(args.sway_min_coef), float(args.sway_max_coef))
        return {
            "adaptive_sway_mode": mode,
            "severity_id": severity_id,
            "severity_label": sev_label,
            "base_sway_coef": base_coef,
            "sway_sampling_coef_used": coef,
            **rhythm_info,
        }

    raise ValueError(f"Unsupported adaptive_sway_mode: {mode}")


def append_aux_log(output_wav: Path, info: Dict, args: argparse.Namespace) -> None:
    try:
        out_dir = Path(getattr(args, "out_dir", output_wav.parent.parent)).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / str(getattr(args, "rhyss_aux_log_name", "rhyss_aux_generation_log.csv"))
        row = {"gen_audio": str(output_wav), **info}
        fieldnames = list(row.keys())
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[RhySS][WARN] Failed to write auxiliary log for {output_wav}: {e}")


class FullF5TTDSGenerator(base.UnifiedSeverityF5TTSGenerator):
    """
    Severity-aware F5-TTDS generator with Chained CFG and RhySS.
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

        if bool(self.args.chained_cfg) and severity_id is None:
            raise ValueError(
                "chained_cfg=True requires severity_id. "
                "Please use --severity_condition_mode speaker."
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        model_obj = self.get_model(ckpt_file, vocab_file)

        ref_audio_processed, ref_text_processed = self.preprocess_ref_audio_text(
            str(ref_audio),
            ref_text,
        )

        default_sway = (
            float(self.args.sway_sampling_coef)
            if sway_sampling_coef is None
            else float(sway_sampling_coef)
        )
        rhyss_info = resolve_rhyss_sway(
            ref_audio=Path(ref_audio),
            ref_text=ref_text,
            severity_id=severity_id,
            default_sway=default_sway,
            args=self.args,
        )
        local_sway_sampling_coef = float(rhyss_info["sway_sampling_coef_used"])

        print(
            "[Full F5-TTDS] "
            f"mode={rhyss_info['adaptive_sway_mode']} | "
            f"severity_id={severity_id} | severity={rhyss_info['severity_label']} | "
            f"base_sway={float(rhyss_info['base_sway_coef']):.4f} | "
            f"rhythm_score={float(rhyss_info['rhythm_score']) if math.isfinite(float(rhyss_info['rhythm_score'])) else float('nan'):.4f} | "
            f"final_sway={local_sway_sampling_coef:.4f} | "
            f"chained_cfg={bool(self.args.chained_cfg)}"
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
            chained_cfg=bool(self.args.chained_cfg),
            text_cfg_strength=float(self.args.text_cfg_strength),
            speaker_cfg_strength=float(self.args.speaker_cfg_strength),
            severity_cfg_strength=float(self.args.severity_cfg_strength),
            speed=self.args.speed,
            fix_duration=self.args.fix_duration if self.args.fix_duration > 0 else None,
            device=self.args.device,
        )
        elapsed = time.perf_counter() - start

        self.sf.write(str(output_wav), audio_segment, final_sample_rate)

        if bool(self.args.remove_silence):
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
            remove_silence_for_generated_wav(str(output_wav))

        rhyss_info_for_log = dict(rhyss_info)
        rhyss_info_for_log["generation_time_sec"] = float(elapsed)
        rhyss_info_for_log["chained_cfg"] = int(bool(self.args.chained_cfg))
        rhyss_info_for_log["text_cfg_strength"] = float(self.args.text_cfg_strength)
        rhyss_info_for_log["speaker_cfg_strength"] = float(self.args.speaker_cfg_strength)
        rhyss_info_for_log["severity_cfg_strength"] = float(self.args.severity_cfg_strength)
        append_aux_log(output_wav, rhyss_info_for_log, self.args)

        return float(elapsed)


def main():
    # Patch the original unified-eval module without modifying its file.
    base.parse_args = parse_args_full
    base.UnifiedSeverityF5TTSGenerator = FullF5TTDSGenerator
    base.main()


if __name__ == "__main__":
    main()
