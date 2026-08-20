"""Quality check: Stage 2 Kannada->Tulu LoRA (sat_Olck alias) on held-out dev."""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

from tulu_lang_alias import PAIR_DIR, SRC_KN, TULU_LANG_TAG, apply_tulu_processor_override

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE = "./model_cache/indictrans2-en-indic-1B"
DEFAULT_ADAPTER = "./stage2_kn_tcy_output"
DEV_SRC = f"./stage2_data/dev/{PAIR_DIR}/dev.{SRC_KN}"
DEV_TGT = f"./stage2_data/dev/{PAIR_DIR}/dev.{TULU_LANG_TAG}"
N_EVAL = 100
BATCH = 4
OUT_PATH = "stage2_quality_report.txt"
RESULTS_MD_DEFAULT = "./stage2_output/RESULTS_v2.md"


def detect_run_mode(src_file: str) -> tuple[str, str, str]:
    """Infer whether this evaluation is real KN-TCY or synthetic fallback."""
    src_path = str(src_file).replace("\\", "/")
    if "kan_Knda-sat_Olck" in src_path:
        return (
            "real",
            "Real KN-TCY mode: Kannada->Tulu data from kn_tcy_raw.csv (organizer-provided parallel data).",
            SRC_KN,
        )
    if "eng_Latn-sat_Olck" in src_path:
        return (
            "synthetic_degraded",
            "Synthetic degraded fallback mode: English->Tulu continuity data (not a real KN-TCY run).",
            "eng_Latn",
        )
    return (
        "unknown",
        "Unknown mode: source path does not match expected real or synthetic stage2_data layout.",
        SRC_KN,
    )


def write_results_markdown(path: str, mode_key: str, mode_text: str, adapter: str, src: str, tgt: str, n_eval: int, bleu: float, chrf: float) -> None:
    mode_title = {
        "real": "Real KN-TCY run",
        "synthetic_degraded": "Synthetic degraded fallback run",
        "unknown": "Unknown-mode run",
    }[mode_key]
    lines = [
        "# Stage 2 Results v2",
        "",
        "## Run identity",
        f"- Mode: {mode_title}",
        f"- Plain-language mode summary: {mode_text}",
        "- Resume source: stage1_output/checkpoint-1500",
        "- Not resumed from discarded stage2_output/checkpoint-3000 (different non-standard tag setup)",
        f"- Adapter evaluated: {adapter}",
        f"- Direction: {src} -> {tgt} (Tulu via sat_Olck alias)",
        f"- Evaluated samples: {n_eval}",
        "",
        "## Metrics",
        f"- BLEU: {bleu:.2f}",
        f"- chrF: {chrf:.2f}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def resolve_adapter(path: str) -> str:
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    ckpts = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    for c in reversed(ckpts):
        if (c / "adapter_config.json").exists():
            return str(c)
    return str(p)


def load_model(base: str, adapter: str, device: str):
    print(f"Loading base model on {device} ...")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    return tok, model


def translate(sentences, model, tok, ip, src, tgt, device):
    outs = []
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i : i + BATCH]
        batch = ip.preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
        inputs = tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_new_tokens=128,
                num_beams=5,
                num_return_sequences=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        decoded = tok.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        outs += ip.postprocess_batch(decoded, lang=tgt)
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--n_eval", type=int, default=N_EVAL)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--results_md", default=RESULTS_MD_DEFAULT)
    parser.add_argument("--src_file", default=DEV_SRC)
    parser.add_argument("--ref_file", default=DEV_TGT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = resolve_adapter(args.adapter)
    tgt = TULU_LANG_TAG
    mode_key, mode_text, src = detect_run_mode(args.src_file)

    if not Path(args.src_file).exists() or not Path(args.ref_file).exists():
        raise FileNotFoundError(
            f"Missing dev files:\n  {args.src_file}\n  {args.ref_file}\n"
            "Run: python prepare_stage2_data.py --mode real or --mode synthetic_degraded"
        )

    with open(args.src_file, encoding="utf-8") as f:
        srcs = [l.strip() for l in f if l.strip()][: args.n_eval]
    with open(args.ref_file, encoding="utf-8") as f:
        refs = [l.strip() for l in f if l.strip()][: args.n_eval]

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    ip = IndicProcessor(inference=True)
    apply_tulu_processor_override(ip)
    tok, model = load_model(args.base, adapter, device)

    log(f"mode={mode_key}")
    log(f"mode_summary={mode_text}")
    log("resume_source=stage1_output/checkpoint-1500")
    log("not_resumed_from=stage2_output/checkpoint-3000 (discarded non-standard tag run)")
    log(f"adapter={adapter}")
    log(f"direction={src}->{tgt} (Tulu via sat_Olck alias; not real Santali)")
    log(f"n_eval={len(srcs)}")
    log()

    log(f"=== Dev set sample ({len(srcs)}) with Stage 2 LoRA ===")
    preds = translate(srcs, model, tok, ip, src, tgt, device)
    bleu = BLEU().corpus_score(preds, [refs])
    chrf = CHRF().corpus_score(preds, [refs])
    log(f"BLEU: {bleu.score:.2f}")
    log(f"chrF: {chrf.score:.2f}")
    log()

    write_results_markdown(
        path=args.results_md,
        mode_key=mode_key,
        mode_text=mode_text,
        adapter=adapter,
        src=src,
        tgt=tgt,
        n_eval=len(srcs),
        bleu=bleu.score,
        chrf=chrf.score,
    )

    log("--- First 8 Pred vs Ref ---")
    for i in range(min(8, len(srcs))):
        log(f"[{i+1}] SRC:  {srcs[i]}")
        log(f"    PRED: {preds[i]}")
        log(f"    REF:  {refs[i]}")
        log()

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.results_md}")


if __name__ == "__main__":
    main()
