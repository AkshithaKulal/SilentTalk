#!/usr/bin/env python3
"""Predict sign label from video and translate to Tulu/Kannada via IndicTrans2 + TTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Force UTF-8 encoding for console output (handles Kannada/Indic scripts on Windows)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from predict_sign import predict_video_topk

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    IndicProcessor = None

# Paths relative to this file's location (isl_recognition/), going up one level to IndicTrans2/
_HF_INTERFACE = Path(__file__).resolve().parent.parent / "IndicTrans2" / "huggingface_interface"
_BASE_MODEL_PATH = str(_HF_INTERFACE / "model_cache" / "indictrans2-en-indic-1B")
_LORA_CHECKPOINT = str(_HF_INTERFACE / "stage1_output" / "checkpoint-1500")


class IndicTranslator:
    """Wrapper for IndicTrans2 fine-tuned model (base 1B + LoRA adapter from checkpoint-1500)."""

    def __init__(self, target_lang: str = "kan_Knda", device: str | None = None):
        if IndicProcessor is None:
            raise ImportError("IndicTransToolkit not available. Install: pip install IndicTransToolkit")

        self.target_lang = target_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading base model: {_BASE_MODEL_PATH}", file=sys.stderr)
        print(f"Applying LoRA adapter: {_LORA_CHECKPOINT}", file=sys.stderr)

        # Load tokenizer from base model (same as test_inference.py)
        self.tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_PATH, trust_remote_code=True)

        # Load base 1B model then apply fine-tuned LoRA adapter — exact pattern from test_inference.py
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            _BASE_MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
        )
        self.model = PeftModel.from_pretrained(base_model, _LORA_CHECKPOINT)
        self.model.eval()
        self.model = self.model.to(self.device)

        if self.device == "cuda":
            self.model = self.model.half()

        self.processor = IndicProcessor(inference=True)
        print(f"Fine-tuned model loaded on {self.device} (base=1B + LoRA checkpoint-1500)", file=sys.stderr)

    def translate(self, text: str) -> str:
        """Translate English text to target language.

        Args:
            text: English sentence/word

        Returns:
            Translated text in target language script
        """
        if not text.strip():
            return ""

        try:
            batch = self.processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=self.target_lang)
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(
                self.device
            )

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )

            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translation = self.processor.postprocess_batch(decoded, lang=self.target_lang)[0]
            return translation
        except Exception as e:
            print(f"WARNING: Translation failed for '{text}': {e}", file=sys.stderr)
            return ""


def resolve_sentence(
    top_preds: list[tuple[str, float]], translator: IndicTranslator, confidence_threshold: float = 0.5
) -> tuple[str | None, str | None]:
    """Translate top-1 label to target language if confident. Otherwise (None, None)."""
    if not top_preds:
        return None, None

    top_label, top_prob = top_preds[0]

    # Only translate if confidence meets threshold
    if top_prob >= confidence_threshold:
        translated = translator.translate(top_label)
        if translated:
            return top_label, translated

    # Below threshold or translation failed
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


def _speak_mms_kan(sentence: str) -> tuple[bool, str]:
    """Primary TTS: facebook/mms-tts-kan — real Kannada neural voice."""
    try:
        import numpy as np
        import scipy.io.wavfile
        import tempfile
        import sounddevice as sd
        from transformers import VitsModel, AutoTokenizer

        model = VitsModel.from_pretrained("facebook/mms-tts-kan")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")
        model.eval()

        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        audio = waveform.squeeze().numpy()
        audio_norm = (audio / max(np.abs(audio).max(), 1e-6) * 32767).astype(np.int16)
        sr = model.config.sampling_rate

        # Play via sounddevice (no file needed)
        sd.play(audio_norm, samplerate=sr)
        sd.wait()
        return True, "mms-tts-kan (Kannada neural TTS)"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _speak_pyttsx3_fallback(sentence: str) -> tuple[bool, str]:
    """Fallback TTS: pyttsx3 system voice (English only — poor for Kannada)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voice_id, reason = pick_voice_id(engine)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.setProperty("rate", 165)
        engine.say(sentence)
        engine.runAndWait()
        return True, f"pyttsx3 fallback ({reason})"
    except Exception as exc:  # noqa: BLE001
        return False, f"pyttsx3 failure: {exc}"


def speak_sentence(sentence: str) -> tuple[bool, str]:
    """Speak sentence — tries MMS Kannada neural TTS first, falls back to pyttsx3."""
    ok, msg = _speak_mms_kan(sentence)
    if ok:
        return True, msg
    # MMS failed (e.g. sounddevice not installed) — fall back
    print(f"WARNING: MMS TTS failed ({msg}), falling back to pyttsx3", file=sys.stderr)
    return _speak_pyttsx3_fallback(sentence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict sign and translate to Kannada/Tulu via IndicTrans2 + TTS")
    parser.add_argument("--video", type=Path, required=False)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech output")
    parser.add_argument(
        "--demo-text",
        type=str,
        default="",
        help="Translate and speak this sentence directly (skip video prediction)",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="If classifier artifacts are missing, skip to translation fallback",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Only translate if top-1 prediction confidence >= this threshold (default 0.5)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="kan_Knda",
        choices=["kan_Knda", "tul_Knda"],
        help="Target language (kan_Knda=Kannada, tul_Knda=Tulu). Default: kan_Knda",
    )
    args = parser.parse_args()

    # Initialize translator (single load, reused for all predictions)
    try:
        translator = IndicTranslator(target_lang=args.target_lang)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to initialize translator: {exc}", file=sys.stderr)
        return 1

    try:
        if args.demo_text.strip():
            text = args.demo_text.strip()
            print("Mode: demo-text (video prediction skipped)")
            translated = translator.translate(text)
            if not translated:
                print(f"WARNING: Translation failed for '{text}'")
                return 0
            print(f"Input: {text}")
            print(f"Translated: {translated}")

            if args.speak:
                ok, msg = speak_sentence(translated)
                if ok:
                    print(f"TTS: success ({msg})")
                else:
                    print(f"WARNING: TTS skipped/failed ({msg})")
            else:
                print("TTS: disabled (use --speak to enable)")
            return 0

        if not args.video:
            print("ERROR: --video is required unless --demo-text is used.", file=sys.stderr)
            return 1
        if not args.video.exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1

        try:
            top_preds, meta = predict_video_topk(
                video=args.video,
                artifacts=args.artifacts,
                models_dir=Path(__file__).resolve().parent / "models",
                top_k=max(1, args.top_k),
                frame_stride=2,
            )
        except FileNotFoundError as exc:
            if not args.allow_missing_model:
                raise
            print(f"WARNING: {exc}")
            print("Mode: fallback (no classifier, using demo label)")
            fallback_label = "Hello"
            translated = translator.translate(fallback_label)
            if not translated:
                print(f"WARNING: Translation of '{fallback_label}' failed")
                return 0
            print(f"Selected label: {fallback_label}")
            print(f"Translated: {translated}")
            if args.speak:
                ok, msg = speak_sentence(translated)
                if ok:
                    print(f"TTS: success ({msg})")
                else:
                    print(f"WARNING: TTS skipped/failed ({msg})")
            else:
                print("TTS: disabled (use --speak to enable)")
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"frames_kept={meta.get('frames_kept', 'unknown')}")
    print("--- top predictions ---")
    for rank, (label, prob) in enumerate(top_preds, start=1):
        print(f"{rank}. {label:20s}  {prob*100:5.1f}%")

    matched_label, translated = resolve_sentence(top_preds, translator, args.confidence_threshold)
    if not translated:
        top_label = top_preds[0][0] if top_preds else "unknown"
        top_prob = top_preds[0][1] if top_preds else 0.0
        print(
            f"WARNING: Top-1 prediction '{top_label}' ({top_prob*100:.1f}%) is below confidence threshold ({args.confidence_threshold}). Not speaking."
        )
        return 0

    print(f"Selected label: {matched_label}")
    print(f"Translated: {translated}")

    if args.speak:
        ok, msg = speak_sentence(translated)
        if ok:
            print(f"TTS: success ({msg})")
        else:
            print(f"WARNING: TTS skipped/failed ({msg})")
    else:
        print("TTS: disabled (use --speak to enable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
