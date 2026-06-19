#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch generation + objective evaluation for speaker-specific F5-TTS checkpoints on TORGO.

This version is based on the minimal test script that worked for the user:
- It DOES NOT call `f5-tts_infer-cli` for generation.
- It calls F5-TTS internal inference utilities directly.
- It exposes `--use_ema`; default is 0 because EMA=0 sounded closer to the fine-tuned TORGO voice.
- It reads per-speaker F5-TTS metadata.csv automatically, so no global manifest.csv is required.
- It removes WER/CER and downstream ASR metrics.

Expected user-side layout:

[PATH-CHECK-1] F5-TTS project root:
    /home/user/ly/F5-TTS-main

[PATH-CHECK-2] Per-speaker data folders:
    /home/user/ly/F5-TTS-main/data/torgo_f01_pinyin/metadata.csv
    /home/user/ly/F5-TTS-main/data/torgo_f03_pinyin/metadata.csv
    /home/user/ly/F5-TTS-main/data/torgo_m01_pinyin/metadata.csv
    ...

    metadata.csv format should be F5-TTS style without a header:
        wav_name.wav|transcript text

[PATH-CHECK-3] Per-speaker checkpoints:
    /home/user/ly/F5-TTS-main/ckpts/torgo_f01/model_last.pt
    /home/user/ly/F5-TTS-main/ckpts/torgo_f03/model_last.pt
    /home/user/ly/F5-TTS-main/ckpts/torgo_m01/model_last.pt
    ...

    The script maps:
        data/torgo_f01_pinyin -> ckpts/torgo_f01/model_last.pt

[PATH-CHECK-4] Custom vocab:
    If your fine-tuning produced a vocab.txt, keep --auto_vocab 1.
    The script will try:
        data/torgo_f01_pinyin/vocab.txt
        ckpts/torgo_f01/vocab.txt

[PATH-CHECK-5] Output directory:
    generated wavs, logs and metrics are saved under --out_dir.

Main outputs:
    generated_wavs/<speaker>/<utt_id>.wav
    auto_manifest.csv
    generation_log.csv
    per_utterance_metrics.csv
    summary_overall.csv
    summary_by_speaker.csv
    failed_items.csv

Recommended first run:
    python batch_f5tts_torgo_direct_use_ema0_generate_eval.py --max_items 6 --skip_existing 0

After listening to the generated wavs and confirming they are normal, set --max_items -1.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Generic utilities
# -----------------------------


def setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def sanitize_filename(name: str) -> str:
    name = safe_str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    return name[:180] if len(name) > 180 else name


def ensure_file(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def ensure_dir(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def count_words(text: str) -> int:
    text = safe_str(text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return 0 if not text else len(text.split())


def numeric_mean_std(df: pd.DataFrame, group_cols: Optional[List[str]], out_csv: Path) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return
    if group_cols:
        group_cols = [c for c in group_cols if c in df.columns]
    if group_cols:
        summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
    else:
        stats: Dict[str, float] = {}
        for col in numeric_cols:
            values = df[col].values
            stats[f"{col}_mean"] = float(np.nanmean(values)) if len(values) else np.nan
            stats[f"{col}_std"] = float(np.nanstd(values)) if len(values) else np.nan
        stats["count"] = int(len(df))
        summary = pd.DataFrame([stats])
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")


# -----------------------------
# Metadata adapter
# -----------------------------


def infer_speaker_from_dataset_name(dataset_name: str) -> str:
    """Convert torgo_f01_pinyin -> F01."""
    name = safe_str(dataset_name).strip()
    m = re.search(r"torgo[_-]([a-zA-Z]\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    name = re.sub(r"_pinyin$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^torgo[_-]", "", name, flags=re.IGNORECASE)
    return name.upper() if name else dataset_name


def resolve_metadata_audio_path(dataset_dir: Path, wav_name: str, audio_subdir: str = "") -> Path:
    """Resolve wav path from metadata row."""
    raw = Path(safe_str(wav_name).strip())
    if raw.is_absolute():
        return raw

    candidates: List[Path] = []
    if audio_subdir:
        candidates.append(dataset_dir / audio_subdir / raw)
    candidates.extend([
        dataset_dir / raw,
        dataset_dir / "wavs" / raw,
        dataset_dir / "wav" / raw,
        dataset_dir / "audio" / raw,
        dataset_dir / "audios" / raw,
    ])
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve() if candidates else (dataset_dir / raw).resolve()


def read_pipe_metadata(metadata_csv: Path, dataset_dir: Path, audio_subdir: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if "|" not in line:
                logging.warning("Skip malformed metadata line %s:%d | %s", metadata_csv, line_no, line)
                continue
            wav_name, text = line.split("|", 1)
            wav_name = wav_name.strip()
            text = text.strip()
            if not wav_name or not text:
                continue
            audio_path = resolve_metadata_audio_path(dataset_dir, wav_name, audio_subdir)
            rows.append({
                "utt_id": Path(wav_name).stem,
                "wav_name": wav_name,
                "audio": str(audio_path),
                "text": text,
            })
    return rows


def choose_reference_row(rows: List[Dict[str, Any]], strategy: str, ref_index: int, random_seed: int) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot choose reference row from an empty metadata file.")
    if strategy == "first":
        return rows[0]
    if strategy == "by_index":
        return rows[min(max(ref_index, 0), len(rows) - 1)]
    if strategy == "random":
        rng = np.random.default_rng(random_seed if random_seed >= 0 else 42)
        return rows[int(rng.integers(0, len(rows)))]
    if strategy == "self":
        return rows[0]  # handled separately when building rows
    # default: longest transcript; usually more stable than a one-word prompt
    return max(rows, key=lambda r: len(safe_str(r.get("text", "")).split()))


def build_manifest_from_data_root(
    data_root: str,
    speaker_glob: str,
    metadata_name: str,
    audio_subdir: str,
    ref_strategy: str,
    ref_index: int,
    random_seed: int,
    exclude_ref_from_generation: bool,
) -> pd.DataFrame:
    root = ensure_dir(data_root, "data_root")
    dataset_dirs = sorted([p for p in root.glob(speaker_glob) if p.is_dir()])
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders matched: {root / speaker_glob}")

    out_rows: List[Dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        metadata_csv = dataset_dir / metadata_name
        if not metadata_csv.exists():
            logging.warning("metadata.csv not found, skip: %s", dataset_dir)
            continue

        dataset = dataset_dir.name
        speaker = infer_speaker_from_dataset_name(dataset)
        speaker_lower = speaker.lower()
        speaker_upper = speaker.upper()
        dataset_no_pinyin = re.sub(r"_pinyin$", "", dataset, flags=re.IGNORECASE)
        ckpt_dataset = dataset_no_pinyin.lower()  # torgo_f01_pinyin -> torgo_f01

        rows = read_pipe_metadata(metadata_csv, dataset_dir, audio_subdir)
        rows = [r for r in rows if Path(r["audio"]).exists()]
        if not rows:
            logging.warning("No valid wav|text rows with existing audio in %s", metadata_csv)
            continue

        fixed_ref = choose_reference_row(rows, ref_strategy, ref_index, random_seed)

        for item in rows:
            ref = item if ref_strategy == "self" else fixed_ref
            if exclude_ref_from_generation and item["utt_id"] == ref["utt_id"]:
                continue
            out_rows.append({
                "utt_id": f"{speaker}_{item['utt_id']}",
                "speaker": speaker,
                "speaker_lower": speaker_lower,
                "speaker_upper": speaker_upper,
                "dataset": dataset,
                "dataset_no_pinyin": dataset_no_pinyin,
                "ckpt_dataset": ckpt_dataset,
                "dataset_dir": str(dataset_dir.resolve()),
                "metadata_csv": str(metadata_csv.resolve()),
                "ref_audio": ref["audio"],
                "ref_text": ref["text"],
                "gen_text": item["text"],
                "target_audio": item["audio"],
                "target_text": item["text"],
                "source_wav_name": item["wav_name"],
                "ref_utt_id": ref["utt_id"],
                "ref_wav_name": ref["wav_name"],
            })

    if not out_rows:
        raise ValueError("No rows were built from data_root. Check data_root/speaker_glob/metadata/audio paths.")
    return pd.DataFrame(out_rows)


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def format_with_row(template: str, row: Dict[str, Any]) -> str:
    values = SafeDict({k: safe_str(v) for k, v in row.items()})
    return template.format_map(values)


def resolve_ckpt(row: Dict[str, Any], ckpt_root: str, ckpt_pattern: str) -> Path:
    if safe_str(row.get("ckpt_file", "")).strip():
        return ensure_file(row["ckpt_file"], "ckpt_file")
    ckpt_root_p = Path(ckpt_root).expanduser().resolve()
    rel = format_with_row(ckpt_pattern, row)
    return ensure_file(ckpt_root_p / rel, "ckpt_file")


def resolve_vocab(row: Dict[str, Any], ckpt_file: Path, args: argparse.Namespace) -> Optional[Path]:
    # Explicit global vocab has the highest priority.
    if safe_str(args.vocab_file).strip():
        return ensure_file(args.vocab_file, "vocab_file")

    # Optional template, e.g. {dataset}/vocab.txt under data_root or ckpt_root.
    if safe_str(args.vocab_pattern).strip():
        rel = format_with_row(args.vocab_pattern, row)
        candidates = [Path(args.data_root).expanduser().resolve() / rel, Path(args.ckpt_root).expanduser().resolve() / rel]
        for p in candidates:
            if p.exists() and p.is_file():
                return p.resolve()
        raise FileNotFoundError(f"vocab_pattern was set but no file found. Tried: {candidates}")

    if not bool(args.auto_vocab):
        return None

    dataset_dir = Path(safe_str(row.get("dataset_dir", ""))).expanduser()
    candidates = [
        dataset_dir / "vocab.txt",
        dataset_dir / "vocab.json",
        ckpt_file.parent / "vocab.txt",
        ckpt_file.parent / "vocab.json",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    return None


# -----------------------------
# F5-TTS direct generator with EMA switch
# -----------------------------


class DirectF5TTSGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.src_dir = self.project_root / "src"
        if self.src_dir.exists():
            sys.path.insert(0, str(self.src_dir))
        os.chdir(self.project_root)

        try:
            import torch  # noqa: F401
            import soundfile as sf  # noqa: F401
            from hydra.utils import get_class  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
            from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Failed to import F5-TTS direct inference modules. Run inside your F5-TTS environment, "
                "and make sure `pip install -e .` has been done. Original error: " + repr(e)
            )

        # Imports are stored as attributes to avoid repeating import logic.
        import torch
        import soundfile as sf
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text

        self.torch = torch
        self.sf = sf
        self.get_class = get_class
        self.OmegaConf = OmegaConf
        self.infer_process = infer_process
        self.load_model = load_model
        self.load_vocoder = load_vocoder
        self.preprocess_ref_audio_text = preprocess_ref_audio_text

        self.model_cfg_path = self.resolve_model_cfg(args.model_name, args.model_cfg)
        self.model_cfg = self.OmegaConf.load(str(self.model_cfg_path))
        self.model_cls = self.get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
        self.model_arc = self.model_cfg.model.arch

        self.vocoder_name = args.vocoder_name
        try:
            yaml_mel = self.model_cfg.model.mel_spec.mel_spec_type
            if args.model_name != "F5TTS_Base" and self.vocoder_name != yaml_mel:
                logging.info("Model yaml expects vocoder=%s; overriding %s -> %s", yaml_mel, self.vocoder_name, yaml_mel)
                self.vocoder_name = yaml_mel
        except Exception:
            pass

        self.vocoder = self._load_vocoder_once()
        self.current_key: Optional[Tuple[str, str, int, str]] = None
        self.current_model = None

    def resolve_model_cfg(self, model: str, model_cfg_arg: str = "") -> Path:
        if model_cfg_arg:
            return ensure_file(model_cfg_arg, "model_cfg")
        p = self.project_root / "src" / "f5_tts" / "configs" / f"{model}.yaml"
        if p.exists():
            return p.resolve()
        try:
            from importlib.resources import files
            res = files("f5_tts").joinpath(f"configs/{model}.yaml")
            return Path(str(res))
        except Exception as e:
            raise FileNotFoundError(f"Cannot locate config for model={model}. Tried {p}. Error: {e}")

    def _load_vocoder_once(self):
        if self.vocoder_name == "vocos":
            local_path = str(self.project_root / "checkpoints" / "vocos-mel-24khz")
        else:
            local_path = str(self.project_root / "checkpoints" / "bigvgan_v2_24khz_100band_256x")
        return self.load_vocoder(
            vocoder_name=self.vocoder_name,
            is_local=bool(self.args.load_vocoder_from_local),
            local_path=local_path,
            device=self.args.device,
        )

    def _unload_current_model(self) -> None:
        self.current_model = None
        self.current_key = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def get_model(self, ckpt_file: Path, vocab_file: Optional[Path]):
        key = (str(ckpt_file), str(vocab_file) if vocab_file else "", int(self.args.use_ema), self.args.model_name)
        if self.current_key == key and self.current_model is not None:
            return self.current_model

        self._unload_current_model()
        logging.info("Loading model | ckpt=%s | use_ema=%s | vocab=%s", ckpt_file, self.args.use_ema, vocab_file or "<default>")
        self.current_model = self.load_model(
            self.model_cls,
            self.model_arc,
            str(ckpt_file),
            mel_spec_type=self.vocoder_name,
            vocab_file=str(vocab_file) if vocab_file else "",
            device=self.args.device,
            use_ema=bool(self.args.use_ema),
        )
        self.current_key = key
        return self.current_model

    def generate_one(
        self,
        ref_audio: Path,
        ref_text: str,
        gen_text: str,
        ckpt_file: Path,
        vocab_file: Optional[Path],
        output_wav: Path,
    ) -> float:
        if not ref_audio.exists():
            raise FileNotFoundError(f"ref_audio not found: {ref_audio}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        model_obj = self.get_model(ckpt_file, vocab_file)

        ref_audio_processed, ref_text_processed = self.preprocess_ref_audio_text(str(ref_audio), ref_text)
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
            sway_sampling_coef=self.args.sway_sampling_coef,
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


# -----------------------------
# Audio objective metrics
# -----------------------------


def load_audio(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    import librosa
    y, sr2 = librosa.load(str(path), sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y, sr2


def dbfs_from_rms(rms: float, eps: float = 1e-10) -> float:
    return 20.0 * math.log10(max(float(rms), eps))


def basic_audio_stats(y: np.ndarray, sr: int) -> Dict[str, float]:
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    return {
        "duration_sec": duration,
        "rms_dbfs": dbfs_from_rms(rms),
        "peak_dbfs": 20.0 * math.log10(peak),
        "clipping_ratio": clipping_ratio,
    }


def frame_rms_db(y: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> Tuple[np.ndarray, int, int]:
    import librosa
    frame_length = max(16, int(sr * frame_ms / 1000.0))
    hop_length = max(8, int(sr * hop_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return db.astype(np.float32), frame_length, hop_length


def pause_metrics(y: np.ndarray, sr: int, top_db: float, frame_ms: float, hop_ms: float, min_pause_ms: float) -> Dict[str, float]:
    db, _, hop_length = frame_rms_db(y, sr, frame_ms, hop_ms)
    total_duration = len(y) / sr
    if db.size == 0 or total_duration <= 0:
        return {"pause_ratio": np.nan, "pause_duration_sec": np.nan, "speech_duration_sec": np.nan, "n_pauses": np.nan, "mean_pause_duration_sec": np.nan}
    threshold = float(np.max(db) - top_db)
    speech_mask = db > threshold
    silence_mask = ~speech_mask
    hop_sec = hop_length / sr
    pause_duration = float(np.sum(silence_mask) * hop_sec)
    speech_duration = max(total_duration - pause_duration, 1e-8)
    min_pause_frames = max(1, int(round((min_pause_ms / 1000.0) / hop_sec)))
    pauses: List[int] = []
    cur = 0
    for v in silence_mask.tolist():
        if v:
            cur += 1
        else:
            if cur >= min_pause_frames:
                pauses.append(cur)
            cur = 0
    if cur >= min_pause_frames:
        pauses.append(cur)
    pause_durs = [p * hop_sec for p in pauses]
    return {
        "pause_ratio": float(np.clip(pause_duration / max(total_duration, 1e-8), 0.0, 1.0)),
        "pause_duration_sec": pause_duration,
        "speech_duration_sec": speech_duration,
        "n_pauses": float(len(pauses)),
        "mean_pause_duration_sec": float(np.mean(pause_durs)) if pause_durs else 0.0,
    }


def extract_prosody(y: np.ndarray, sr: int, frame_ms: float, hop_ms: float, fmin: float, fmax: float, top_db: float) -> Dict[str, Any]:
    import librosa
    db, frame_length, hop_length = frame_rms_db(y, sr, frame_ms, hop_ms)
    threshold = float(np.max(db) - top_db) if db.size else -80.0
    pause_mask = (db <= threshold).astype(np.float32)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=max(1024, int(2 ** math.ceil(math.log2(max(frame_length, 16))))),
            hop_length=hop_length,
        )
        f0 = np.asarray(f0, dtype=np.float32)
        voiced_flag = np.asarray(voiced_flag, dtype=bool)
    except Exception:
        f0 = np.full_like(db, np.nan, dtype=np.float32)
        voiced_flag = np.zeros_like(db, dtype=bool)

    finite_f0 = f0[np.isfinite(f0)]
    stats = {
        "f0_mean_hz": float(np.mean(finite_f0)) if finite_f0.size else np.nan,
        "f0_std_hz": float(np.std(finite_f0)) if finite_f0.size else np.nan,
        "f0_range_hz": float(np.max(finite_f0) - np.min(finite_f0)) if finite_f0.size else np.nan,
        "voiced_ratio": float(np.mean(voiced_flag)) if voiced_flag.size else np.nan,
        "energy_db_mean": float(np.mean(db)) if db.size else np.nan,
        "energy_db_std": float(np.std(db)) if db.size else np.nan,
    }
    return {"f0": f0, "energy_db": db, "pause_mask": pause_mask, "stats": stats}


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x, dtype=np.float32)
    if np.sum(good) == 1:
        return np.full_like(x, float(x[good][0]), dtype=np.float32)
    return np.interp(idx, idx[good], x[good]).astype(np.float32)


def _resample_curve(x: np.ndarray, length: int = 200) -> np.ndarray:
    x = _interp_nan(x)
    if x.size == 0:
        return np.zeros(length, dtype=np.float32)
    if x.size == 1:
        return np.full(length, float(x[0]), dtype=np.float32)
    old = np.linspace(0.0, 1.0, num=x.size)
    new = np.linspace(0.0, 1.0, num=length)
    return np.interp(new, old, x).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != b.size or a.size < 3:
        return np.nan
    if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def autopcp_proxy(ref_prosody: Dict[str, Any], gen_prosody: Dict[str, Any]) -> Dict[str, float]:
    ref_f0 = _resample_curve(ref_prosody["f0"])
    gen_f0 = _resample_curve(gen_prosody["f0"])
    ref_energy = _resample_curve(ref_prosody["energy_db"])
    gen_energy = _resample_curve(gen_prosody["energy_db"])
    ref_pause = _resample_curve(ref_prosody["pause_mask"])
    gen_pause = _resample_curve(gen_prosody["pause_mask"])
    f0_corr = _pearson(ref_f0, gen_f0)
    energy_corr = _pearson(ref_energy, gen_energy)
    pause_corr = _pearson(ref_pause, gen_pause)
    mapped = [((float(c) + 1.0) / 2.0) for c in [f0_corr, energy_corr, pause_corr] if np.isfinite(c)]
    return {
        "autopcp_proxy": float(np.mean(mapped)) if mapped else np.nan,
        "f0_corr_ref_gen": f0_corr,
        "energy_corr_ref_gen": energy_corr,
        "pause_corr_ref_gen": pause_corr,
    }


class SpeakerSimilarity:
    def __init__(self, enable: bool, backend: str, model_name: str, device: str, sr: int = 16000):
        self.enable = bool(enable)
        self.backend = backend.lower()
        self.model_name = model_name
        self.device = device
        self.sr = sr
        self.model = None
        if not self.enable or self.backend == "none":
            self.enable = False
            return
        if self.backend != "speechbrain":
            raise ValueError("Only --sim_backend speechbrain or none is supported.")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            run_opts = {"device": device} if device else {}
            self.model = EncoderClassifier.from_hparams(source=model_name, run_opts=run_opts)
        except Exception as e:
            logging.warning("Failed to initialize speaker similarity model. SIM-o will be NaN. Error: %s", e)
            self.enable = False

    def _embed(self, wav_path: Path) -> Optional[np.ndarray]:
        if not self.enable or self.model is None:
            return None
        try:
            import librosa
            import torch
            wav, _ = librosa.load(str(wav_path), sr=self.sr, mono=True)
            if wav.size == 0:
                return None
            wav_t = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                emb = self.model.encode_batch(wav_t)
            return emb.squeeze().detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            logging.warning("Speaker embedding failed for %s: %s", wav_path, e)
            return None

    def sim_o(self, ref_audio: Path, gen_audio: Path) -> float:
        e1 = self._embed(ref_audio)
        e2 = self._embed(gen_audio)
        if e1 is None or e2 is None:
            return np.nan
        denom = float(np.linalg.norm(e1) * np.linalg.norm(e2))
        if denom <= 1e-8:
            return np.nan
        return float(np.dot(e1, e2) / denom)


def evaluate_row(row: Dict[str, Any], gen_audio: Path, args: argparse.Namespace, sim_model: SpeakerSimilarity) -> Dict[str, Any]:
    result: Dict[str, Any] = {k: row.get(k, "") for k in row.keys()}
    result["gen_audio"] = str(gen_audio)
    result["eval_status"] = "ok"

    ref_audio = Path(safe_str(row.get("ref_audio", ""))).expanduser().resolve()
    gen_text = safe_str(row.get("gen_text", ""))

    y_gen, sr = load_audio(gen_audio, args.sr)
    gen_basic = basic_audio_stats(y_gen, sr)
    gen_pause = pause_metrics(y_gen, sr, args.pause_top_db, args.frame_ms, args.hop_ms, args.min_pause_ms)
    gen_prosody = extract_prosody(y_gen, sr, args.frame_ms, args.hop_ms, args.f0_min, args.f0_max, args.pause_top_db)

    result.update({f"gen_{k}": v for k, v in gen_basic.items()})
    result.update({f"gen_{k}": v for k, v in gen_pause.items()})
    result.update({f"gen_{k}": v for k, v in gen_prosody["stats"].items()})

    n_words = count_words(gen_text)
    result["n_words_gen_text"] = float(n_words)
    result["speech_rate_wps"] = float(n_words / max(gen_pause["speech_duration_sec"], 1e-8)) if n_words else np.nan
    result["overall_word_rate_wps"] = float(n_words / max(gen_basic["duration_sec"], 1e-8)) if n_words else np.nan

    if ref_audio.exists():
        y_ref, _ = load_audio(ref_audio, args.sr)
        ref_basic = basic_audio_stats(y_ref, args.sr)
        ref_pause = pause_metrics(y_ref, args.sr, args.pause_top_db, args.frame_ms, args.hop_ms, args.min_pause_ms)
        ref_prosody = extract_prosody(y_ref, args.sr, args.frame_ms, args.hop_ms, args.f0_min, args.f0_max, args.pause_top_db)
        result.update({f"ref_{k}": v for k, v in ref_basic.items()})
        result.update({f"ref_{k}": v for k, v in ref_pause.items()})
        result.update({f"ref_{k}": v for k, v in ref_prosody["stats"].items()})
        result["duration_ratio_syn_ref"] = float(gen_basic["duration_sec"] / max(ref_basic["duration_sec"], 1e-8))
        result["pause_ratio_diff_syn_ref"] = float(gen_pause["pause_ratio"] - ref_pause["pause_ratio"])
        result["abs_pause_ratio_diff_syn_ref"] = float(abs(result["pause_ratio_diff_syn_ref"]))
        result.update(autopcp_proxy(ref_prosody, gen_prosody))
        result["sim_o"] = sim_model.sim_o(ref_audio, gen_audio) if sim_model.enable else np.nan
    else:
        result["duration_ratio_syn_ref"] = np.nan
        result["pause_ratio_diff_syn_ref"] = np.nan
        result["abs_pause_ratio_diff_syn_ref"] = np.nan
        result["autopcp_proxy"] = np.nan
        result["f0_corr_ref_gen"] = np.nan
        result["energy_corr_ref_gen"] = np.nan
        result["pause_corr_ref_gen"] = np.nan
        result["sim_o"] = np.nan

    try:
        gen_time = float(row.get("generation_time_sec", np.nan))
    except Exception:
        gen_time = np.nan
    result["rtf_generation"] = float(gen_time / max(gen_basic["duration_sec"], 1e-8)) if np.isfinite(gen_time) and gen_time > 0 else np.nan
    return result


# -----------------------------
# Output helpers
# -----------------------------


def write_metric_readme(path: Path) -> None:
    text = """# F5-TTS TORGO direct generation objective evaluation

This experiment uses speaker-specific fine-tuned F5-TTS checkpoints to generate TORGO dysarthric speech.

Important settings:
- Direct internal F5-TTS inference is used instead of `f5-tts_infer-cli`.
- `use_ema=0` is the default because the user's minimal test showed it better preserved the fine-tuned speaker voice.
- WER/CER and downstream ASR metrics are intentionally excluded.

Core metrics:
- `sim_o`: speaker embedding cosine similarity between reference prompt and generated speech.
- `autopcp_proxy`: F0/energy/pause-profile correlation proxy for prosodic consistency.
- `gen_pause_ratio`: pause duration ratio of generated speech.
- `duration_ratio_syn_ref`: generated duration / reference-prompt duration.
- `speech_rate_wps`: word count / generated effective speech duration.
- `overall_word_rate_wps`: word count / generated total duration.
- `rtf_generation`: generation time / generated duration.
- Auxiliary acoustic metrics include RMS, peak, clipping ratio, F0 statistics, voiced ratio, and energy statistics.
"""
    path.write_text(text, encoding="utf-8")


def write_rating_sheet(df: pd.DataFrame, out_csv: Path) -> None:
    cols = [c for c in ["utt_id", "speaker", "ref_audio", "gen_audio", "gen_text"] if c in df.columns]
    rating = df[cols].copy() if cols else pd.DataFrame()
    rating["mos_naturalness_1_5"] = ""
    rating["smos_similarity_1_5"] = ""
    rating["comments"] = ""
    rating.to_csv(out_csv, index=False, encoding="utf-8-sig")


# -----------------------------
# Args and main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch F5-TTS TORGO generation/evaluation with explicit use_ema control.")

    # [PATH-CHECK-1] F5-TTS project root.
    p.add_argument("--project_root", default="/home/user/ly/F5-TTS-main", help="F5-TTS repository root.")
    # [PATH-CHECK-2] Root containing torgo_f01_pinyin, torgo_m01_pinyin, etc.
    p.add_argument("--data_root", default="/home/user/ly/F5-TTS-main/data", help="Root directory containing per-speaker F5-TTS data folders.")
    p.add_argument("--speaker_glob", default="torgo_*_pinyin", help="Glob pattern for speaker folders under data_root.")
    p.add_argument("--metadata_name", default="metadata.csv")
    # [PATH-CHECK-3] If wavs are under each folder's wavs/ subdir, set --audio_subdir wavs. Otherwise leave empty.
    p.add_argument("--audio_subdir", default="")
    # Optional manifest; generally not needed for your current folder layout.
    p.add_argument("--manifest", default="")
    p.add_argument("--save_auto_manifest", type=int, default=1)

    # Reference selection.
    p.add_argument("--ref_strategy", default="longest_text", choices=["first", "longest_text", "by_index", "random", "self"])
    p.add_argument("--ref_index", type=int, default=0)
    p.add_argument("--exclude_ref_from_generation", type=int, default=1)

    # [PATH-CHECK-4] Checkpoint layout.
    p.add_argument("--ckpt_root", default="/home/user/ly/F5-TTS-main/ckpts", help="Root dir of fine-tuned checkpoints.")
    p.add_argument("--ckpt_pattern", default="{ckpt_dataset}/model_last.pt", help="Path pattern under ckpt_root. Current default maps torgo_f01_pinyin -> torgo_f01/model_last.pt")

    # [PATH-CHECK-5] Vocab. Leave --vocab_file empty and --auto_vocab 1 unless you know a fixed vocab path.
    p.add_argument("--vocab_file", default="")
    p.add_argument("--vocab_pattern", default="", help="Optional vocab path pattern, e.g. {dataset}/vocab.txt. Usually not needed.")
    p.add_argument("--auto_vocab", type=int, default=1)

    # [PATH-CHECK-6] Output directory.
    p.add_argument("--out_dir", default="/home/user/ly/F5-TTS-main/exp/f5tts_torgo_direct_use_ema0_eval")

    # F5-TTS model settings.
    # IMPORTANT: this must match the base model used for fine-tuning.
    p.add_argument("--model_name", default="F5TTS_v1_Base", help="Use the same value that worked in the minimal script: F5TTS_v1_Base or F5TTS_Base.")
    p.add_argument("--model_cfg", default="")
    p.add_argument("--vocoder_name", default="vocos", choices=["vocos", "bigvgan"])
    p.add_argument("--load_vocoder_from_local", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # IMPORTANT: default 0 because user confirmed EMA=0 is better for fine-tuned TORGO models.
    p.add_argument("--use_ema", type=int, default=0)
    p.add_argument("--nfe_step", type=int, default=32)
    p.add_argument("--cfg_strength", type=float, default=2.0)
    p.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--cross_fade_duration", type=float, default=0.15)
    p.add_argument("--fix_duration", type=float, default=-1.0)
    p.add_argument("--remove_silence", type=int, default=0, help="Keep 0 for dysarthric rhythm evaluation.")

    # Run control.
    p.add_argument("--skip_existing", type=int, default=1)
    p.add_argument("--overwrite_existing", type=int, default=0, help="If 1, regenerate even when wav exists. Overrides skip_existing.")
    p.add_argument("--eval_only", type=int, default=0)
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--max_items", type=int, default=-1, help="Debugging: use a small number first; -1 means all.")
    p.add_argument("--seed", type=int, default=-1)

    # Objective metrics.
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


def validate_manifest(df: pd.DataFrame, eval_only: bool) -> None:
    required = ["utt_id", "speaker", "ref_audio", "ref_text", "gen_text"]
    if eval_only:
        required.append("gen_audio")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}; existing columns: {list(df.columns)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    gen_dir = out_dir / "generated_wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(out_dir / "run.log")

    logging.info("[PATH-CHECK-1] project_root = %s", args.project_root)
    logging.info("[PATH-CHECK-2] data_root    = %s", args.data_root)
    logging.info("[PATH-CHECK-3] audio_subdir = %s", args.audio_subdir if args.audio_subdir else "<auto/direct>")
    logging.info("[PATH-CHECK-4] ckpt_root   = %s", args.ckpt_root)
    logging.info("[PATH-CHECK-4] ckpt_pattern= %s", args.ckpt_pattern)
    logging.info("[PATH-CHECK-5] auto_vocab  = %s | vocab_file=%s | vocab_pattern=%s", args.auto_vocab, args.vocab_file or "<auto>", args.vocab_pattern or "<none>")
    logging.info("[PATH-CHECK-6] out_dir     = %s", out_dir)
    logging.info("[INFER] model_name=%s | use_ema=%s | nfe=%s | cfg=%s | sway=%s | remove_silence=%s", args.model_name, args.use_ema, args.nfe_step, args.cfg_strength, args.sway_sampling_coef, args.remove_silence)

    if args.manifest.strip():
        df = pd.read_csv(ensure_file(args.manifest, "manifest"))
        manifest_source = args.manifest
    else:
        df = build_manifest_from_data_root(
            data_root=args.data_root,
            speaker_glob=args.speaker_glob,
            metadata_name=args.metadata_name,
            audio_subdir=args.audio_subdir,
            ref_strategy=args.ref_strategy,
            ref_index=args.ref_index,
            random_seed=args.seed if args.seed >= 0 else 42,
            exclude_ref_from_generation=bool(args.exclude_ref_from_generation),
        )
        manifest_source = f"auto_from_data_root:{args.data_root}"
        if args.save_auto_manifest:
            df.to_csv(out_dir / "auto_manifest.csv", index=False, encoding="utf-8-sig")
            logging.info("Saved auto manifest: %s", out_dir / "auto_manifest.csv")
            logging.info("Auto manifest preview:\n%s", df.head(5).to_string(index=False))

    validate_manifest(df, bool(args.eval_only))
    if args.max_items is not None and args.max_items > 0:
        df = df.head(args.max_items).copy()
        logging.info("Debug mode: only first %d items will be used.", args.max_items)

    config = vars(args).copy()
    config["manifest_source"] = manifest_source
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metric_readme(out_dir / "README_metrics.md")

    if args.dry_run:
        # Print expected key paths without loading model.
        dry_rows = []
        for _, row_s in df.iterrows():
            row = row_s.to_dict()
            try:
                ckpt = resolve_ckpt(row, args.ckpt_root, args.ckpt_pattern)
                vocab = resolve_vocab(row, ckpt, args)
                out_wav = gen_dir / sanitize_filename(row.get("speaker", "unknown")) / f"{sanitize_filename(row.get('utt_id', 'utt'))}.wav"
                dry_rows.append({**row, "ckpt_file": str(ckpt), "vocab_file": str(vocab) if vocab else "", "output_wav": str(out_wav), "dry_status": "ok"})
            except Exception as e:
                dry_rows.append({**row, "dry_status": "failed", "dry_error": str(e)})
        pd.DataFrame(dry_rows).to_csv(out_dir / "dry_run_preview.csv", index=False, encoding="utf-8-sig")
        logging.info("Dry run finished. See %s", out_dir / "dry_run_preview.csv")
        return

    generation_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []

    # Stage 1: generation
    if not bool(args.eval_only):
        generator = DirectF5TTSGenerator(args)
        for _, row_s in df.iterrows():
            row = row_s.to_dict()
            utt_id = sanitize_filename(row.get("utt_id", "utt"))
            speaker = sanitize_filename(row.get("speaker", "unknown_speaker"))
            speaker_dir = gen_dir / speaker
            output_wav = speaker_dir / f"{utt_id}.wav"
            rec: Dict[str, Any] = {"utt_id": row.get("utt_id", utt_id), "speaker": row.get("speaker", ""), "gen_audio": str(output_wav)}
            try:
                if bool(args.skip_existing) and not bool(args.overwrite_existing) and output_wav.exists() and output_wav.stat().st_size > 0:
                    rec.update({"generation_status": "skipped_existing", "generation_time_sec": 0.0, "generation_returncode": 0})
                    logging.info("Skip existing: %s", output_wav)
                else:
                    ckpt_file = resolve_ckpt(row, args.ckpt_root, args.ckpt_pattern)
                    vocab_file = resolve_vocab(row, ckpt_file, args)
                    ref_audio = ensure_file(row["ref_audio"], "ref_audio")
                    ref_text = safe_str(row["ref_text"]).strip()
                    gen_text = safe_str(row["gen_text"]).strip()
                    if not ref_text or not gen_text:
                        raise ValueError("ref_text/gen_text is empty. Check metadata.csv.")
                    logging.info("Generating %s | speaker=%s | ckpt=%s | use_ema=%s", utt_id, speaker, ckpt_file, args.use_ema)
                    elapsed = generator.generate_one(ref_audio, ref_text, gen_text, ckpt_file, vocab_file, output_wav)
                    rec.update({
                        "generation_status": "ok",
                        "generation_time_sec": float(elapsed),
                        "generation_returncode": 0,
                        "ckpt_file": str(ckpt_file),
                        "vocab_file": str(vocab_file) if vocab_file else "",
                        "use_ema": int(args.use_ema),
                    })
            except Exception as e:
                logging.exception("Generation failed: utt_id=%s", row.get("utt_id", ""))
                rec.update({"generation_status": "exception", "generation_error": str(e), "generation_returncode": 1, "generation_time_sec": np.nan})
                failed_records.append({**row, **rec})
            generation_records.append(rec)

        gen_df = pd.DataFrame(generation_records)
        gen_df.to_csv(out_dir / "generation_log.csv", index=False, encoding="utf-8-sig")
        df = df.merge(gen_df[["utt_id", "gen_audio", "generation_status", "generation_time_sec"]], on="utt_id", how="left")
    else:
        df["generation_status"] = "eval_only"
        if "generation_time_sec" not in df.columns:
            df["generation_time_sec"] = np.nan

    # Stage 2: objective evaluation
    sim_model = SpeakerSimilarity(enable=bool(args.enable_sim), backend=args.sim_backend, model_name=args.sim_model, device=args.device)
    eval_records: List[Dict[str, Any]] = []
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


if __name__ == "__main__":
    main()
