#!/usr/bin/env python3
"""SilentTalk Flask backend — ISL sign prediction + Kannada translation + TTS."""

import sys, io, base64, tempfile, json, threading
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
ISL_DIR     = ROOT / "isl_recognition"
ARTIFACTS   = ISL_DIR / "transfer_pack"
MODELS_DIR  = ISL_DIR / "models"
SAMPLES_DIR = ISL_DIR / "verification_set"
HF_BASE     = ROOT / "IndicTrans2" / "huggingface_interface"

# Base 1B model: local model_cache → HuggingFace Hub (auto-downloads on first use)
_LOCAL_BASE = HF_BASE / "model_cache" / "indictrans2-en-indic-1B"
BASE_MODEL_PATH = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "ai4bharat/indictrans2-en-indic-1B"

# LoRA checkpoint: local stage1_output → downloaded checkpoint-1500-inference/ at root
_LOCAL_LORA      = HF_BASE / "stage1_output" / "checkpoint-1500"
_DOWNLOADED_LORA = ROOT / "checkpoint-1500-inference"
LORA_PATH = (
    str(_LOCAL_LORA) if _LOCAL_LORA.exists()
    else str(_DOWNLOADED_LORA) if _DOWNLOADED_LORA.exists()
    else None
)

sys.path.insert(0, str(ISL_DIR))

# ── app ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── lazy-loaded models ────────────────────────────────────────────────────────
_classifier   = None
_label_enc    = None
_translator   = None
_tts_model    = None
_tts_tokenizer = None
_mediapipe_ready = False
_load_lock    = threading.Lock()


def get_classifier():
    global _classifier, _label_enc
    if _classifier is None:
        import joblib
        _classifier = joblib.load(ARTIFACTS / "sign_classifier.joblib")
        _label_enc  = joblib.load(ARTIFACTS / "label_encoder.joblib")
    return _classifier, _label_enc


def get_translator():
    global _translator
    if _translator is None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        if LORA_PATH is None:
            raise FileNotFoundError("LoRA checkpoint not found. Run setup.ps1 to auto-download it.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, attn_implementation="eager")
        model = PeftModel.from_pretrained(base, LORA_PATH).to(device)
        if device == "cuda":
            model = model.half()
        model.eval()
        ip = IndicProcessor(inference=True)
        _translator = (model, tokenizer, ip, device)
    return _translator


def get_tts():
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        from transformers import VitsModel, AutoTokenizer as AT
        _tts_model = VitsModel.from_pretrained("facebook/mms-tts-kan")
        _tts_tokenizer = AT.from_pretrained("facebook/mms-tts-kan")
        _tts_model.eval()
    return _tts_model, _tts_tokenizer


def translate_text(text: str) -> str:
    import torch
    model, tokenizer, ip, device = get_translator()
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, num_beams=5,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang="kan_Knda")[0]


def synthesize_wav(text: str) -> bytes:
    import torch, scipy.io.wavfile
    model, tokenizer = get_tts()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().numpy()
    audio = (audio / max(np.abs(audio).max(), 1e-6) * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, model.config.sampling_rate, audio)
    return buf.getvalue()


def predict_frame_sequence(frames_bgr: list) -> list[tuple[str, float]]:
    """Run MediaPipe + classifier on a list of BGR frames, return top-5."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from extract_landmarks import frame_feature, FEAT_DIM, download_if_missing, HAND_MODEL_URL, POSE_MODEL_URL
    from train_classifier import sequence_to_features

    hand_model = download_if_missing(HAND_MODEL_URL, MODELS_DIR / "hand_landmarker.task")
    pose_model = download_if_missing(POSE_MODEL_URL, MODELS_DIR / "pose_landmarker_lite.task")

    BaseOptions = mp_python.BaseOptions
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)

    feats = []
    with vision.PoseLandmarker.create_from_options(pose_opts) as pl, \
         vision.HandLandmarker.create_from_options(hand_opts) as hl:
        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = i * 33
            pr = pl.detect_for_video(mp_img, ts)
            hr = hl.detect_for_video(mp_img, ts)
            feats.append(frame_feature(pr, hr))

    if not feats:
        return []

    seq = np.stack(feats)
    x = sequence_to_features(seq).reshape(1, -1)
    clf, le = get_classifier()
    proba = clf.predict_proba(x)[0]
    idx = np.argsort(proba)[::-1][:5]
    return [(le.classes_[i], float(proba[i])) for i in idx]


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/api/signs")
def api_signs():
    signs = []
    if SAMPLES_DIR.exists():
        for folder in sorted(SAMPLES_DIR.iterdir()):
            if folder.is_dir():
                import re
                label = re.sub(r"^\d+\.\s*", "", folder.name).strip()
                vids = sorted([v for v in folder.iterdir() if v.suffix.lower() in (".mp4", ".mov", ".avi")])
                if vids:
                    signs.append({"label": label, "folder": folder.name, "sample": vids[0].name})
    return jsonify(signs)


@app.route("/")
def index():
    return send_from_directory(ROOT / "static" / "react", "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ROOT / "static" / "react" / "assets", filename)


@app.route("/sample/<folder>/<filename>")
def serve_sample(folder, filename):
    return send_from_directory(SAMPLES_DIR / folder, filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    frames_b64 = data.get("frames", [])
    if not frames_b64:
        return jsonify({"error": "No frames provided"}), 400

    frames_bgr = []
    for b64 in frames_b64:
        img_bytes = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames_bgr.append(frame)

    if not frames_bgr:
        return jsonify({"error": "Could not decode frames"}), 400

    try:
        preds = predict_frame_sequence(frames_bgr)
        if not preds:
            return jsonify({"error": "No predictions"}), 500
        top_label, top_conf = preds[0]
        return jsonify({
            "top_label": top_label,
            "top_conf": round(top_conf * 100, 1),
            "top5": [{"label": l, "conf": round(c * 100, 1)} for l, c in preds]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        translation = translate_text(text)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        wav_bytes = synthesize_wav(text)
        b64 = base64.b64encode(wav_bytes).decode()
        return jsonify({"audio_b64": b64, "format": "wav"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    clf_ok = (ARTIFACTS / "sign_classifier.joblib").exists()
    model_ok = (HF_BASE / "model_cache" / "indictrans2-en-indic-1B").exists() or True  # falls back to HF Hub
    lora_ok = (HF_BASE / "stage1_output" / "checkpoint-1500" / "adapter_model.safetensors").exists()
    mms_ok = Path.home().joinpath(".cache/huggingface/hub/models--facebook--mms-tts-kan").exists()
    return jsonify({
        "classifier": clf_ok,
        "translation_model": model_ok,
        "lora_adapter": lora_ok,
        "tts_model": mms_ok,
    })


if __name__ == "__main__":
    print("Starting SilentTalk server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
