#!/usr/bin/env python3
"""Batch video processor for ISL sign prediction with comprehensive logging and metrics."""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Force UTF-8 encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from predict_sign import predict_video_topk


class IndicTranslator:
    """Wrapper for IndicTrans2 English→Indic translation."""

    def __init__(self, target_lang: str = "kan_Knda", device: str | None = None):
        self.target_lang = target_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "ai4bharat/indictrans2-en-indic-dist-200M"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="eager"
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.processor = IndicProcessor(inference=True)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            batch = self.processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=self.target_lang)
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(self.device)
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
            return ""


class BatchVideoProcessor:
    """Process multiple videos and log predictions."""

    def __init__(self, artifacts_path: Path, output_dir: Path = None, confidence_threshold: float = 0.5, target_lang: str = "kan_Knda"):
        self.artifacts_path = Path(artifacts_path)
        self.output_dir = Path(output_dir or Path.cwd() / "batch_results")
        self.output_dir.mkdir(exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.target_lang = target_lang
        self.translator = IndicTranslator(target_lang=target_lang)

        # Results storage
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def process_video(self, video_path: Path, ground_truth_label: Optional[str] = None) -> dict:
        """Process single video and return prediction."""
        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video not found: {video_path}"}

        try:
            print(f"Processing: {video_path.name}")

            # Predict
            top_preds, meta = predict_video_topk(
                video=video_path,
                artifacts=self.artifacts_path,
                models_dir=self.artifacts_path.parent / "models",
                top_k=5,
                frame_stride=2,
            )

            top_label, top_prob = top_preds[0] if top_preds else ("unknown", 0.0)

            # Translate
            translated = self.translator.translate(top_label) if top_prob >= self.confidence_threshold else ""

            # Result
            result = {
                "video": video_path.name,
                "timestamp": datetime.now().isoformat(),
                "frames_kept": meta.get("frames_kept", 0),
                "predicted_label": top_label,
                "confidence": float(top_prob),
                "meets_threshold": top_prob >= self.confidence_threshold,
                "translated": translated,
                "ground_truth": ground_truth_label or "unknown",
                "correct": (top_label.lower() == ground_truth_label.lower()) if ground_truth_label else None,
                "top_5_predictions": [(label, float(prob)) for label, prob in top_preds],
            }

            self.results.append(result)
            self._print_result(result)
            return result

        except Exception as e:
            error_result = {"video": video_path.name, "error": str(e)}
            self.results.append(error_result)
            print(f"  ERROR: {e}")
            return error_result

    def _print_result(self, result: dict):
        """Print formatted result."""
        if "error" in result:
            print(f"  ERROR: {result['error']}\n")
            return

        print(f"  Predicted: {result['predicted_label']} ({result['confidence']*100:.1f}%)")
        print(f"  Frames: {result['frames_kept']}")
        print(f"  Translated: {result['translated']}")
        if result["ground_truth"] != "unknown":
            status = "✓ CORRECT" if result["correct"] else "✗ WRONG"
            print(f"  Ground Truth: {result['ground_truth']} {status}")
        print()

    def process_batch(self, video_files: list[tuple[Path, Optional[str]]]):
        """Process multiple videos."""
        print(f"\n{'='*60}")
        print(f"BATCH VIDEO PROCESSING - {len(video_files)} videos")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Target Language: {self.target_lang}")
        print(f"{'='*60}\n")

        for video_path, ground_truth in video_files:
            self.process_video(video_path, ground_truth)

        self._save_results()

    def _save_results(self):
        """Save results to CSV and JSON."""
        # CSV
        csv_path = self.output_dir / f"batch_results_{self.timestamp}.csv"
        if self.results and "error" not in self.results[0]:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "video",
                    "predicted_label",
                    "confidence",
                    "meets_threshold",
                    "translated",
                    "ground_truth",
                    "correct",
                    "frames_kept",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writeheader()
                for r in self.results:
                    if "error" not in r:
                        row = {k: r.get(k, "") for k in fieldnames}
                        writer.writerow(row)
            print(f"Saved: {csv_path}")

        # JSON
        json_path = self.output_dir / f"batch_results_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # Summary
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics."""
        valid_results = [r for r in self.results if "error" not in r]
        if not valid_results:
            print("\nNo valid results to summarize.")
            return

        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")

        total = len(valid_results)
        print(f"Total Videos: {total}")

        # Confidence statistics
        confidences = [r["confidence"] for r in valid_results]
        print(f"Avg Confidence: {sum(confidences) / len(confidences):.3f}")
        print(f"Min Confidence: {min(confidences):.3f}")
        print(f"Max Confidence: {max(confidences):.3f}")

        # Threshold pass rate
        passed = sum(1 for r in valid_results if r["meets_threshold"])
        print(f"Passed Threshold ({self.confidence_threshold}): {passed}/{total} ({passed*100//total}%)")

        # Accuracy (if ground truth available)
        with_gt = [r for r in valid_results if r["ground_truth"] != "unknown"]
        if with_gt:
            correct = sum(1 for r in with_gt if r["correct"])
            print(f"Accuracy (ground truth): {correct}/{len(with_gt)} ({correct*100//len(with_gt)}%)")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process videos for sign prediction")
    parser.add_argument("--videos", nargs="+", required=True, help="Video file paths")
    parser.add_argument("--ground-truth", nargs="+", help="Ground truth labels (same order as videos)")
    parser.add_argument("--artifacts", type=Path, default=Path("transfer_pack"), help="Artifacts directory")
    parser.add_argument("--output", type=Path, help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--lang", default="kan_Knda", choices=["kan_Knda", "tul_Knda"], help="Target language")

    args = parser.parse_args()

    # Prepare video list
    video_list = list(zip([Path(v) for v in args.videos], args.ground_truth or [None] * len(args.videos)))

    # Process
    processor = BatchVideoProcessor(
        artifacts_path=args.artifacts,
        output_dir=args.output,
        confidence_threshold=args.threshold,
        target_lang=args.lang,
    )
    processor.process_batch(video_list)
