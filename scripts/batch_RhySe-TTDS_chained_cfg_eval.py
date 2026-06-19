#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chained-CFG batch inference/evaluation for F5-TTDS.

This script does not replace batch_f5ttds_severity_unified_eval.py.
It reuses the already tested unified severity-aware inference pipeline,
but patches the generator so that model.sample() receives:

    chained_cfg=True
    text_cfg_strength
    speaker_cfg_strength
    severity_cfg_strength

Expected chain:
    batch_f5ttds_chained_cfg_eval.py
        -> batch_f5ttds_severity_unified_eval.main()
        -> ChainedCFGGenerator.generate_one()
        -> infer_process(..., chained_cfg=True, ...)
        -> CFM.sample(..., chained_cfg=True, ...)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import batch_f5ttds_severity_unified_eval as base


_ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args_chained():
    """
    Parse chained-CFG-specific arguments first, then pass all remaining args
    to the original severity-unified evaluation parser.
    """
    chained_parser = argparse.ArgumentParser(add_help=False)
    chained_parser.add_argument(
        "--chained_cfg",
        type=int,
        default=1,
        help="1: use four-branch chained CFG; 0: fall back to ordinary CFG.",
    )
    chained_parser.add_argument(
        "--text_cfg_strength",
        type=float,
        default=1.5,
        help="Guidance scale for v_text - v_empty.",
    )
    chained_parser.add_argument(
        "--speaker_cfg_strength",
        type=float,
        default=1.0,
        help="Guidance scale for v_text_spk - v_text.",
    )
    chained_parser.add_argument(
        "--severity_cfg_strength",
        type=float,
        default=0.5,
        help="Guidance scale for v_text_spk_sev - v_text_spk.",
    )

    chained_args, remaining_argv = chained_parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining_argv
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = old_argv

    args.chained_cfg = bool(chained_args.chained_cfg)
    args.text_cfg_strength = float(chained_args.text_cfg_strength)
    args.speaker_cfg_strength = float(chained_args.speaker_cfg_strength)
    args.severity_cfg_strength = float(chained_args.severity_cfg_strength)

    return args


class ChainedCFGGenerator(base.UnifiedSeverityF5TTSGenerator):
    """
    Same as UnifiedSeverityF5TTSGenerator, but passes chained-CFG arguments
    into infer_process().
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

        return float(elapsed)


def main():
    # Patch the original unified-eval module without modifying its file.
    base.parse_args = parse_args_chained
    base.UnifiedSeverityF5TTSGenerator = ChainedCFGGenerator
    base.main()


if __name__ == "__main__":
    main()