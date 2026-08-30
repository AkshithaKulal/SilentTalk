import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

HF_BASE = Path('IndicTrans2/huggingface_interface')
words = ['Sunday', 'Neighbour', 'Hello', 'Mother', 'Hospital', 'Thank you', 'Friend', 'School', 'Market', 'Monday']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ip = IndicProcessor(inference=True)


def translate_batch(model, tokenizer, texts):
    results = []
    for text in texts:
        batch = ip.preprocess_batch([text], src_lang='eng_Latn', tgt_lang='kan_Knda')
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding='longest').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, num_beams=5)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append(ip.postprocess_batch(decoded, lang='kan_Knda')[0])
    return results


print('Loading OLD model (dist-200M raw)...', file=sys.stderr)
tok_old = AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True)
mdl_old = AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M', trust_remote_code=True, attn_implementation='eager').to(device)
if device == 'cuda':
    mdl_old = mdl_old.half()
mdl_old.eval()
old_results = translate_batch(mdl_old, tok_old, words)
del mdl_old
torch.cuda.empty_cache()

print('Loading NEW model (1B base + LoRA checkpoint-1500)...', file=sys.stderr)
tok_new = AutoTokenizer.from_pretrained(str(HF_BASE / 'model_cache/indictrans2-en-indic-1B'), trust_remote_code=True)
base = AutoModelForSeq2SeqLM.from_pretrained(str(HF_BASE / 'model_cache/indictrans2-en-indic-1B'), trust_remote_code=True, attn_implementation='eager')
mdl_new = PeftModel.from_pretrained(base, str(HF_BASE / 'stage1_output/checkpoint-1500')).to(device)
if device == 'cuda':
    mdl_new = mdl_new.half()
mdl_new.eval()
new_results = translate_batch(mdl_new, tok_new, words)

lines = []
lines.append(f"{'English':<20} {'OLD (dist-200M)':<35} {'NEW (checkpoint-1500)':<35} SAME?")
lines.append('-' * 100)
for word, old, new in zip(words, old_results, new_results):
    same = 'SAME' if old.strip('.,') == new.strip('.,') else 'DIFFERENT'
    lines.append(f'{word:<20} {old:<35} {new:<35} {same}')

output = '\n'.join(lines)
print(output)

# Write to file for external reading
out_path = Path('test_results/model_comparison.txt')
out_path.write_text(output, encoding='utf-8')
print(f'\nSaved to {out_path}', file=sys.stderr)
