from huggingface_hub import login, snapshot_download
import os

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("Set HF_TOKEN env var first, then rerun.")

login(token=token)

snapshot_download(
    repo_id="ai4bharat/indictrans2-en-indic-1B",
    local_dir="./model_cache/indictrans2-en-indic-1B",
    max_workers=2,
)
print("Model downloaded")
