#!/usr/bin/env python3
"""SilentTalk — FastAPI backend.
ISL sign prediction → Kannada translation → TTS (indic-parler-tts).

Run:  uvicorn app:app --host 0.0.0.0 --port 5000
"""

import asyncio, base64, io, json, re, sys, threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── stdout utf-8 ─────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
ISL_DIR     = ROOT / "isl_recognition"
ARTIFACTS   = ISL_DIR / "transfer_pack"
MODELS_DIR  = ISL_DIR / "models"
SAMPLES_DIR = ISL_DIR / "verification_set"
HF_BASE     = ROOT / "IndicTrans2" / "huggingface_interface"

_LOCAL_BASE      = HF_BASE / "model_cache" / "indictrans2-en-indic-1B"
BASE_MODEL_PATH  = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "ai4bharat/indictrans2-en-indic-1B"

_LOCAL_LORA      = HF_BASE / "stage1_output" / "checkpoint-1500"
_DOWNLOADED_LORA = ROOT / "checkpoint-1500-inference"
LORA_PATH = (
    str(_LOCAL_LORA)      if _LOCAL_LORA.exists()
    else str(_DOWNLOADED_LORA) if _DOWNLOADED_LORA.exists()
    else None
)

sys.path.insert(0, str(ISL_DIR))

# ── HuggingFace auth ──────────────────────────────────────────────────────────
# Token stored in .env or environment variable — never hardcode in source
import os
# Load .env file if present (local dev)
_env_file = ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    if _HF_TOKEN:
        from huggingface_hub import login as hf_login
        hf_login(token=_HF_TOKEN, add_to_git_credential=False)
except Exception:
    pass  # offline — cached models still work

# ── model state ───────────────────────────────────────────────────────────────
_classifier      = None
_label_enc       = None
_translator      = None
_tts_model       = None
_tts_tokenizer   = None
_pose_landmarker = None
_hand_landmarker = None
_predict_lock    = threading.Lock()   # MediaPipe is not thread-safe

# ── voice presets ─────────────────────────────────────────────────────────────
VOICE_PRESETS = {
    "female_clear": "Ananya speaks in Kannada with a clear, natural, expressive female voice at a moderate pace. The recording is clean with no background noise.",
    "female_warm":  "Kavitha speaks in Kannada with a warm, gentle, soft-spoken female voice at a slow, deliberate pace. The recording is very high quality.",
    "male_clear":   "Suresh speaks in Kannada with a clear, confident, mid-pitched male voice at a moderate pace. The recording is clean and professional.",
    "male_deep":    "Ramesh speaks in Kannada with a deep, authoritative male voice at a slow, clear pace. High quality studio recording.",
    "neutral":      "A speaker with a neutral, clear Kannada voice at a moderate pace. Clean recording with no noise.",
}
DEFAULT_VOICE = "female_clear"


# ── model loaders ─────────────────────────────────────────────────────────────
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
            raise RuntimeError("LoRA checkpoint not found.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

        if device == "cuda":
            base = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL_PATH, trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.float16, device_map="cuda",
            )
        else:
            base = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL_PATH, trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)

        model = PeftModel.from_pretrained(base, LORA_PATH)
        model.eval()
        _translator = (model, tokenizer, IndicProcessor(inference=True), device)
    return _translator


def get_tts():
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        parler_cached = Path.home().joinpath(
            ".cache/huggingface/hub/models--ai4bharat--indic-parler-tts-pretrained").exists()

        if parler_cached:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer as AT
            model_id = "ai4bharat/indic-parler-tts-pretrained"
            _tts_tokenizer = AT.from_pretrained(model_id)
            _tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            _tts_model._tts_type = "parler"
        else:
            from transformers import VitsModel, AutoTokenizer as AT
            _tts_model = VitsModel.from_pretrained("facebook/mms-tts-kan").to(device)
            _tts_tokenizer = AT.from_pretrained("facebook/mms-tts-kan")
            _tts_model._tts_type = "mms"

        _tts_model.eval()
    return _tts_model, _tts_tokenizer


def _init_mediapipe():
    global _pose_landmarker, _hand_landmarker
    if _pose_landmarker is not None:
        return
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from extract_landmarks import download_if_missing, HAND_MODEL_URL, POSE_MODEL_URL

    hand_model = download_if_missing(HAND_MODEL_URL, MODELS_DIR / "hand_landmarker.task")
    pose_model = download_if_missing(POSE_MODEL_URL, MODELS_DIR / "pose_landmarker_lite.task")
    BaseOptions = mp_python.BaseOptions

    _pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    _hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))


# ── inference helpers ─────────────────────────────────────────────────────────
def predict_frame_sequence(frames_bgr: list) -> list[tuple[str, float]]:
    import mediapipe as mp
    from extract_landmarks import frame_feature
    from train_classifier import sequence_to_features

    _init_mediapipe()

    feats = []
    with _predict_lock:
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pr = _pose_landmarker.detect(mp_img)
            hr = _hand_landmarker.detect(mp_img)
            feats.append(frame_feature(pr, hr))

    if not feats:
        return []

    x = sequence_to_features(np.stack(feats)).reshape(1, -1)
    clf, le = get_classifier()
    proba = clf.predict_proba(x)[0]
    idx = np.argsort(proba)[::-1][:5]
    return [(le.classes_[i], float(proba[i])) for i in idx]


def translate_text(text: str) -> str:
    import torch
    model, tokenizer, ip, device = get_translator()
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding="longest").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, num_beams=5,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang="kan_Knda")[0]


def synthesize_wav(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    import torch, scipy.io.wavfile
    model, tokenizer = get_tts()
    device = next(model.parameters()).device
    tts_type = getattr(model, "_tts_type", "mms")

    if tts_type == "parler":
        desc       = VOICE_PRESETS.get(voice, VOICE_PRESETS[DEFAULT_VOICE])
        input_ids  = tokenizer(desc, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = gen.cpu().numpy().squeeze().astype(np.float32)
    else:
        inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
        with torch.no_grad():
            audio = model(**inputs).waveform.squeeze().cpu().numpy().astype(np.float32)

    sample_rate = model.config.sampling_rate
    audio_i16   = (audio / max(np.abs(audio).max(), 1e-6) * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, audio_i16)
    return buf.getvalue()


# ── lifespan: preload fast models at startup ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    print("▶ Preloading classifier + MediaPipe...", flush=True)
    await loop.run_in_executor(None, get_classifier)
    await loop.run_in_executor(None, _init_mediapipe)
    print("▶ Preloading TTS...", flush=True)
    await loop.run_in_executor(None, get_tts)
    print("✓ All models ready — server is live", flush=True)
    yield
    # shutdown: nothing to clean up


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SilentTalk API", version="2.0.0", lifespan=lifespan)


# ── request/response models ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    frames: list[str]

class TranslateRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    clf_ok  = (ARTIFACTS / "sign_classifier.joblib").exists()
    lora_ok = (_LOCAL_LORA / "adapter_model.safetensors").exists() or \
              (_DOWNLOADED_LORA / "adapter_model.safetensors").exists()
    model_ok = _LOCAL_BASE.exists() or True   # HF Hub fallback always available
    tts_ok   = Path.home().joinpath(
        ".cache/huggingface/hub/models--ai4bharat--indic-parler-tts-pretrained").exists() or \
               Path.home().joinpath(
        ".cache/huggingface/hub/models--facebook--mms-tts-kan").exists()
    return {
        "classifier":       clf_ok,
        "translation_model": model_ok,
        "lora_adapter":     lora_ok,
        "tts_model":        tts_ok,
    }


@app.get("/api/signs")
async def api_signs():
    signs = []
    if SAMPLES_DIR.exists():
        for folder in sorted(SAMPLES_DIR.iterdir()):
            if not folder.is_dir():
                continue
            label = re.sub(r"^\d+\.\s*", "", folder.name).strip()
            vids  = sorted([v for v in folder.iterdir()
                            if v.suffix.lower() in (".mp4", ".mov", ".avi")])
            if vids:
                signs.append({"label": label, "folder": folder.name,
                               "sample": vids[0].name})
    return signs


@app.get("/api/voices")
async def api_voices():
    parler_ready = Path.home().joinpath(
        ".cache/huggingface/hub/models--ai4bharat--indic-parler-tts-pretrained").exists()
    return {
        "voices": [
            {"id": "female_clear", "name": "Ananya",  "description": "Female · Clear · Moderate pace"},
            {"id": "female_warm",  "name": "Kavitha", "description": "Female · Warm · Slow & gentle"},
            {"id": "male_clear",   "name": "Suresh",  "description": "Male · Clear · Moderate pace"},
            {"id": "male_deep",    "name": "Ramesh",  "description": "Male · Deep · Authoritative"},
            {"id": "neutral",      "name": "Neutral", "description": "Neutral · Clear · Standard"},
        ],
        "default": DEFAULT_VOICE,
        "engine": "indic-parler-tts" if parler_ready else "mms-tts-kan",
        "parler_ready": parler_ready,
    }


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    if not req.frames:
        raise HTTPException(400, "No frames provided")

    frames_bgr = []
    for b64 in req.frames:
        raw   = base64.b64decode(b64.split(",")[-1])
        arr   = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames_bgr.append(frame)

    if not frames_bgr:
        raise HTTPException(400, "Could not decode frames")

    loop  = asyncio.get_event_loop()
    preds = await loop.run_in_executor(None, predict_frame_sequence, frames_bgr)

    if not preds:
        raise HTTPException(500, "No predictions returned")

    top_label, top_conf = preds[0]
    return {
        "top_label": top_label,
        "top_conf":  round(top_conf * 100, 1),
        "top5": [{"label": l, "conf": round(c * 100, 1)} for l, c in preds],
    }


@app.post("/api/translate")
async def api_translate(req: TranslateRequest):
    if not req.text:
        raise HTTPException(400, "No text provided")
    loop        = asyncio.get_event_loop()
    translation = await loop.run_in_executor(None, translate_text, req.text)
    return {"translation": translation}


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    if not req.text:
        raise HTTPException(400, "No text provided")
    loop      = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(
        None, synthesize_wav, req.text, req.voice or DEFAULT_VOICE)
    b64 = base64.b64encode(wav_bytes).decode()
    return {"audio_b64": b64, "format": "wav"}


@app.get("/sample/{folder}/{filename}")
async def serve_sample(folder: str, filename: str):
    path = SAMPLES_DIR / folder / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


# ── serve React SPA ───────────────────────────────────────────────────────────
REACT_DIR = ROOT / "static" / "react"
if REACT_DIR.exists():
    app.mount("/assets", StaticFiles(directory=REACT_DIR / "assets"), name="assets")

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    """Serve React index.html for all non-API routes."""
    index = REACT_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm run build"}, 404)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting SilentTalk (FastAPI) at http://localhost:5000")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False, workers=1)
