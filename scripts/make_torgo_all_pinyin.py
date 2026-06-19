#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import shutil
from pathlib import Path

import torchaudio
from datasets import Dataset

PROJECT_ROOT = Path("/home/user/ly/F5-TTS-main")
DATA_ROOT = PROJECT_ROOT / "data"

SRC_SPEAKERS = [
    "torgo_f01_pinyin",
    "torgo_f03_pinyin",
    "torgo_f04_pinyin",
    "torgo_m01_pinyin",
    "torgo_m02_pinyin",
    "torgo_m04_pinyin",
    "torgo_m05_pinyin",
]

OUT_DIR = DATA_ROOT / "torgo_all_pinyin"
OUT_WAV_DIR = OUT_DIR / "wavs"

SEVERITY_MAP = {
    "F01": 1,  # moderate
    "F03": 0,  # low
    "F04": 0,  # low
    "M01": 2,  # high
    "M02": 2,  # high
    "M04": 2,  # high
    "M05": 1,  # moderate
}


def infer_speaker(folder_name: str) -> str:
    m = re.search(r"torgo[_-]([a-zA-Z]\d+)", folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer speaker from folder: {folder_name}")
    return m.group(1).upper()


def find_audio(src_dir: Path, wav_name: str) -> Path:
    candidates = [
        src_dir / "wavs" / wav_name,
        src_dir / wav_name,
        src_dir / "wav" / wav_name,
        src_dir / "audio" / wav_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find audio {wav_name} under {src_dir}")


def get_duration(audio_path: Path) -> float:
    info = torchaudio.info(str(audio_path))
    return float(info.num_frames / info.sample_rate)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata_lines = []
    durations = []

    for folder in SRC_SPEAKERS:
        src_dir = DATA_ROOT / folder
        metadata_path = src_dir / "metadata.csv"

        if not metadata_path.exists():
            print(f"[WARN] missing metadata: {metadata_path}")
            continue

        speaker = infer_speaker(folder)
        severity = SEVERITY_MAP[speaker]

        with metadata_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue

                wav_name, text = line.split("|", 1)
                wav_name = wav_name.strip()
                text = text.strip()

                if not wav_name or not text:
                    continue

                src_audio = find_audio(src_dir, wav_name)

                new_wav_name = f"{speaker}_{Path(wav_name).name}"
                dst_audio = OUT_WAV_DIR / new_wav_name

                if not dst_audio.exists():
                    shutil.copy2(src_audio, dst_audio)

                duration = get_duration(dst_audio)

                rows.append(
                    {
                        "audio_path": str(dst_audio),
                        "text": text,
                        "duration": duration,
                        "speaker": speaker,
                        "severity": severity,
                    }
                )

                durations.append(duration)
                metadata_lines.append(f"wavs/{new_wav_name}|{text}")

    if not rows:
        raise RuntimeError("No rows collected. Please check source folders.")

    # 保存 HuggingFace Dataset，供 CustomDatasetPath 或 CustomDataset 读取
    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(OUT_DIR / "raw"))

    # 同时保存 raw.arrow，兼容你当前 load_dataset 的 fallback 逻辑
    dataset.data.table.combine_chunks().to_batches()

    with (OUT_DIR / "duration.json").open("w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False, indent=2)

    with (OUT_DIR / "metadata.csv").open("w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")

    # vocab.txt 优先复制已有 speaker 的 vocab
    for folder in SRC_SPEAKERS:
        vocab_path = DATA_ROOT / folder / "vocab.txt"
        if vocab_path.exists():
            shutil.copy2(vocab_path, OUT_DIR / "vocab.txt")
            print(f"Copied vocab from {vocab_path}")
            break

    print(f"Done. Total utterances: {len(rows)}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()