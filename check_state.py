from pathlib import Path

checks = {
    "sign_classifier.joblib":    Path("isl_recognition/transfer_pack/sign_classifier.joblib"),
    "label_encoder.joblib":      Path("isl_recognition/transfer_pack/label_encoder.joblib"),
    "hand_landmarker.task":      Path("isl_recognition/models/hand_landmarker.task"),
    "pose_landmarker.task":      Path("isl_recognition/models/pose_landmarker_lite.task"),
    "LoRA checkpoint":           Path("checkpoint-1500-inference/adapter_model.safetensors"),
    "React build":               Path("static/react/index.html"),
    "indic-parler-tts (gated)":  Path.home()/".cache/huggingface/hub/models--ai4bharat--indic-parler-tts-pretrained",
    "mms-tts-kan (fallback)":    Path.home()/".cache/huggingface/hub/models--facebook--mms-tts-kan",
    "indictrans2-1B (HF cache)": Path.home()/".cache/huggingface/hub/models--ai4bharat--indictrans2-en-indic-1B",
}

print("=" * 55)
print("  SilentTalk State Check")
print("=" * 55)
for name, p in checks.items():
    status = "OK      " if p.exists() else "MISSING "
    print(f"  {status}  {name}")

print()

import importlib
packages = ["fastapi", "uvicorn", "parler_tts", "mediapipe", "sklearn", "torch", "flask"]
print("  Packages:")
for pkg in packages:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"  OK        {pkg} {ver}")
    except ImportError:
        print(f"  MISSING   {pkg}")

print()
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
    print(f"  Free VRAM: {free:.1f} GB")

print("=" * 55)
