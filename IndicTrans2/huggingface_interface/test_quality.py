"""Quick quality check: base IndicTrans2 + LoRA (checkpoint-1500) on EN->KN."""
import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

BASE = "./model_cache/indictrans2-en-indic-1B"
ADAPTER = "./stage1_output/checkpoint-1500"
SRC, TGT = "eng_Latn", "kan_Knda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EVAL = 50
BATCH = 4
OUT_PATH = "quality_report.txt"

CUSTOM = [
    "Good morning.",
    "How are you?",
    "Where is the railway station?",
    "I need a doctor.",
    "The weather is nice today.",
    "Please help me.",
    "What is your name?",
    "Thank you very much.",
]


def load_model(with_lora: bool):
    print(f"Loading base model on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if with_lora:
        print(f"Loading LoRA adapter: {ADAPTER}")
        model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()
    model.eval()
    return tok, model


def translate(sentences, model, tok, ip):
    outs = []
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i : i + BATCH]
        batch = ip.preprocess_batch(batch, src_lang=SRC, tgt_lang=TGT)
        inputs = tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
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
        outs += ip.postprocess_batch(decoded, lang=TGT)
        del inputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return outs


def main():
    import sys

    # Avoid Windows console UnicodeEncodeError on Kannada
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_path = "quality_report.txt"
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    with open(
        "stage1_data/dev/eng_Latn-kan_Knda/dev.eng_Latn", encoding="utf-8"
    ) as f:
        srcs = [l.strip() for l in f if l.strip()][:N_EVAL]
    with open(
        "stage1_data/dev/eng_Latn-kan_Knda/dev.kan_Knda", encoding="utf-8"
    ) as f:
        refs = [l.strip() for l in f if l.strip()][:N_EVAL]

    ip = IndicProcessor(inference=True)
    tok, model = load_model(with_lora=True)

    log("=== Custom sentences (LoRA) ===")
    custom_out = translate(CUSTOM, model, tok, ip)
    for s, t in zip(CUSTOM, custom_out):
        log(f"EN: {s}")
        log(f"KN: {t}")
        log()

    log(f"=== Dev set sample ({N_EVAL}) with LoRA ===")
    preds = translate(srcs, model, tok, ip)
    bleu = BLEU().corpus_score(preds, [refs])
    chrf = CHRF().corpus_score(preds, [refs])
    log(f"BLEU: {bleu.score:.2f}")
    log(f"chrF: {chrf.score:.2f}")
    log()

    log("--- First 8 Pred vs Ref ---")
    for i in range(min(8, len(srcs))):
        log(f"[{i+1}] EN:  {srcs[i]}")
        log(f"    PRED: {preds[i]}")
        log(f"    REF:  {refs[i]}")
        log()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
