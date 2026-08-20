# Stage 2 inference: Kannada -> Tulu (sat_Olck alias).
# Run from IndicTrans2/huggingface_interface/
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

from tulu_lang_alias import SRC_KN, TULU_LANG_TAG, apply_tulu_processor_override

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE = "./model_cache/indictrans2-en-indic-1B"
DEFAULT_ADAPTER = "./stage2_kn_tcy_output"
OUT_PATH = "stage2_inference_report.txt"

SAMPLE_KN = [
    "ನಮಸ್ಕಾರ",
    "ಧನ್ಯವಾದಗಳು",
    "ದಯವಿಟ್ಟು ಸಹಾಯ ಮಾಡಿ",
    "ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ",
    "ನನ್ನ ಹೆಸರೇನು",
    "ನಾನು ಹಸಿವಾಗಿದ್ದೇನೆ",
    "ಬನ್ನಿ",
    "ಕ್ಷಮಿಸಿ",
]


def resolve_adapter(path: str) -> str:
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    # prefer highest checkpoint-* if root has none
    ckpts = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    for c in reversed(ckpts):
        if (c / "adapter_config.json").exists():
            return str(c)
    return str(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    adapter = resolve_adapter(args.adapter)
    SRC, TGT = SRC_KN, TULU_LANG_TAG

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base, trust_remote_code=True, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()

    ip = IndicProcessor(inference=True)
    apply_tulu_processor_override(ip)

    print(f"Device: {device}")
    print(f"Adapter: {adapter}")
    print(f"Direction: {SRC} -> {TGT} (Tulu alias; not real Santali)")
    print("-" * 60)

    results = [
        f"adapter={adapter}",
        f"src={SRC} tgt={TGT} (Tulu via sat_Olck alias)",
        "-" * 40,
    ]
    for sentence in SAMPLE_KN:
        batch = ip.preprocess_batch([sentence], src_lang=SRC, tgt_lang=TGT)
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=5,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translation = ip.postprocess_batch(decoded, lang=TGT)[0]
        line = f"KN: {sentence}\nTCY: {translation}"
        print(line)
        print()
        results.append(line)

    Path(args.out).write_text("\n\n".join(results) + "\n", encoding="utf-8")
    print("-" * 60)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
