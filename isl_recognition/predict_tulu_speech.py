#!/usr/bin/env python3
"""Predict sign label from video and speak mapped Tulu sentence (Kannada script)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from predict_sign import predict_video_topk


def load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mapping JSON must be an object of label->sentence")
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key.strip()] = value.strip()
    return out


def resolve_sentence(top_preds: list[tuple[str, float]], mapping: dict[str, str]) -> tuple[str | None, str | None]:
    for label, _prob in top_preds:
        sent = mapping.get(label)
        if sent:
            return label, sent
    return None, None


def pick_voice_id(engine) -> tuple[str | None, str]:
    voices = engine.getProperty("voices") or []
    ranked: list[tuple[int, object]] = []
    for v in voices:
        name = str(getattr(v, "name", "")).lower()
        vid = str(getattr(v, "id", "")).lower()
        langs = getattr(v, "languages", []) or []
        lang_text = " ".join(str(x).lower() for x in langs)
        blob = " ".join([name, vid, lang_text])
        is_kn = (
            "kannada" in blob
            or "kn-in" in blob
            or " b'kn" in blob
            or "[kn" in blob
        )
        is_hi = (
            "hindi" in blob
            or "hi-in" in blob
            or " b'hi" in blob
            or "[hi" in blob
        )
        if is_kn:
            score = 0
        elif is_hi:
            score = 1
        elif "india" in blob or "indic" in blob:
            score = 2
        else:
            score = 3
        ranked.append((score, v))

    if not ranked:
        return None, "no voices reported by engine"

    ranked.sort(key=lambda x: x[0])
    chosen = ranked[0][1]
    voice_id = str(getattr(chosen, "id", "")) or None
    reason = f"voice={getattr(chosen, 'name', 'unknown')} score={ranked[0][0]}"
    return voice_id, reason


def speak_sentence(sentence: str) -> tuple[bool, str]:
    try:
        import pyttsx3
    except Exception as exc:  # noqa: BLE001
        return False, f"pyttsx3 unavailable: {exc}"

    try:
        engine = pyttsx3.init()
        voice_id, reason = pick_voice_id(engine)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.setProperty("rate", 165)
        engine.say(sentence)
        engine.runAndWait()
        return True, reason
    except Exception as exc:  # noqa: BLE001
        return False, f"TTS failure: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict sign and speak mapped Tulu sentence")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "tulu_sentence_map_kn.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech output")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    try:
        mapping = load_mapping(args.mapping)
        top_preds, meta = predict_video_topk(
            video=args.video,
            artifacts=args.artifacts,
            models_dir=Path(__file__).resolve().parent / "models",
            top_k=max(1, args.top_k),
            frame_stride=2,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"frames_kept={meta.get('frames_kept', 'unknown')}")
    print("--- top predictions ---")
    for rank, (label, prob) in enumerate(top_preds, start=1):
        print(f"{rank}. {label:20s}  {prob*100:5.1f}%")

    matched_label, sentence = resolve_sentence(top_preds, mapping)
    if not sentence:
        print("WARNING: No mapped Tulu sentence found in top-k predictions.")
        return 0

    print(f"Selected label: {matched_label}")
    print(f"Spoken sentence: {sentence}")

    if args.speak:
        ok, msg = speak_sentence(sentence)
        if ok:
            print(f"TTS: success ({msg})")
        else:
            print(f"WARNING: TTS skipped/failed ({msg})")
    else:
        print("TTS: disabled (use --speak to enable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
