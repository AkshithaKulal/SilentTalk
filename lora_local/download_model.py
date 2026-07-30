from huggingface_hub import login, snapshot_download

login(token="YOUR_HF_TOKEN")  # paste your token, then delete this line after running once

snapshot_download(
    repo_id="ai4bharat/indictrans2-en-indic-1B",
    local_dir="./model_cache/indictrans2-en-indic-1B",
    max_workers=2,
)
print("Model downloaded")
