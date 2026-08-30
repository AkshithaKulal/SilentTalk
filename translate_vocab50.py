"""Translate all 50 target vocabulary words using checkpoint-1500 fine-tuned model."""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

HF_BASE = Path("IndicTrans2/huggingface_interface")
BASE_MODEL = str(HF_BASE / "model_cache/indictrans2-en-indic-1B")
LORA_CKPT = str(HF_BASE / "stage1_output/checkpoint-1500")

VOCAB_50 = [
    "Hello", "Thank you", "Please", "Sorry", "Yes", "No", "Good", "Bye",
    "I", "You", "He", "She", "Friend", "Doctor", "Teacher",
    "Help", "Water", "Food", "Hungry", "Thirsty",
    "Pain", "Tired", "Happy", "Sad", "Sick",
    "What", "Where", "When", "Who", "How", "Why", "How much", "Which",
    "Home", "School", "Hospital", "Bathroom", "Market", "Bus stop",
    "Today", "Tomorrow", "Yesterday", "Now", "Later", "Time",
    "Go", "Come", "Eat", "Drink", "Sit", "Stop",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base model: {BASE_MODEL}", file=sys.stderr)
print(f"Applying LoRA adapter: {LORA_CKPT}", file=sys.stderr)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, trust_remote_code=True, attn_implementation="eager")
model = PeftModel.from_pretrained(base, LORA_CKPT).to(device)
if device == "cuda":
    model = model.half()
model.eval()
ip = IndicProcessor(inference=True)
print(f"Model ready on {device}\n", file=sys.stderr)

results = []
for word in VOCAB_50:
    batch = ip.preprocess_batch([word], src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, num_beams=5,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    translation = ip.postprocess_batch(decoded, lang="kan_Knda")[0]
    results.append((word, translation))

# Print table
print(f"{'#':<4} {'English':<20} {'Kannada (checkpoint-1500)'}")
print("-" * 60)
for i, (word, trans) in enumerate(results, 1):
    print(f"{i:<4} {word:<20} {trans}")

# Save to file for reviewer
out_path = Path("test_results/vocab50_translations.txt")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"{'#':<4} {'English':<20} {'Kannada (checkpoint-1500)'}\n")
    f.write("-" * 60 + "\n")
    for i, (word, trans) in enumerate(results, 1):
        f.write(f"{i:<4} {word:<20} {trans}\n")
print(f"\nSaved to {out_path}", file=sys.stderr)
