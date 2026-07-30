from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_path = "./model_cache/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load the Tulu monolingual corpus
with open("tulu_wiki_sentences_ready.txt", encoding="utf-8") as f:
    tulu_sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(tulu_sentences)} Tulu sentences")

# Subsample — you don't need all 103K for this, a strong subset is enough
# and keeps translation time reasonable
import random
random.seed(42)
sample_size = 20000  # adjust based on how much time you have
tulu_sample = random.sample(tulu_sentences, min(sample_size, len(tulu_sentences)))

def backtranslate_batch(sentences, batch_size=16):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        # Tag as kan_Knda since the tokenizer doesn't know tcy_Knda —
        # Tulu-in-Kannada-script segments reasonably under the Kannada tokenizer
        tagged = [f"kan_Knda eng_Latn {s}" for s in batch]
        inputs = tokenizer(tagged, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

        if (i // batch_size) % 50 == 0:
            print(f"Processed {i}/{len(sentences)}")

    return results

print("Starting backtranslation...")
english_translations = backtranslate_batch(tulu_sample)

# Save as synthetic parallel corpus: English (synthetic) <-> Tulu (real)
with open("synthetic_en_tulu.csv", "w", encoding="utf-8") as f:
    f.write("english,tulu\n")
    for eng, tulu in zip(english_translations, tulu_sample):
        eng_clean = eng.replace(",", " ").replace("\n", " ")
        tulu_clean = tulu.replace(",", " ").replace("\n", " ")
        f.write(f"{eng_clean},{tulu_clean}\n")

print(f"Saved {len(english_translations)} synthetic pairs to synthetic_en_tulu.csv")
