# Domain inference test for Stage 1 (ISL-style short phrases).
# Run from IndicTrans2/huggingface_interface/
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base_model_path = "./model_cache/indictrans2-en-indic-1B"
checkpoint_path = "./stage1_output/checkpoint-1500"
SRC, TGT = "eng_Latn", "kan_Knda"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_path, trust_remote_code=True, attn_implementation="eager"
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if device == "cuda":
    model = model.half()

ip = IndicProcessor(inference=True)

# Your actual short-sign / phrase vocabulary — real Stage 1 signal
test_sentences = [
    "Hello",
    "Thank you",
    "Please",
    "Sorry",
    "I am hungry",
    "Where is the hospital",
    "I need help",
    "What is your name",
    "I am tired",
    "Come here",
]

print(f"Device: {device}")
print(f"Adapter: {checkpoint_path}")
print("-" * 60)

results = []
for sentence in test_sentences:
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
    line = f"{sentence:30s} -> {translation}"
    print(line)
    results.append(line)

with open("stage1_inference_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results) + "\n")
print("-" * 60)
print("Wrote stage1_inference_report.txt")
