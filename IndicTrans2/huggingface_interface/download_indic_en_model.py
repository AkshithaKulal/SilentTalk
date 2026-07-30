from huggingface_hub import snapshot_download

# You'll need to request access to this one too on huggingface.co, same as before
snapshot_download(
    repo_id="ai4bharat/indictrans2-indic-en-1B",
    local_dir="./model_cache/indictrans2-indic-en-1B",
    max_workers=2,
)
print("Indic-En model downloaded")
