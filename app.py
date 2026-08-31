#!/usr/bin/env python3
"""SilentTalk — FastAPI backend (sign → translate → TTS).

Development (UI + API separate — use this while editing the frontend):
  Terminal 1:  python app.py
  Terminal 2:  cd frontend && npm run dev
  Open:        http://localhost:5173

Production / demo (one port, built UI):
  cd frontend && npm run build
  python app.py
  Open:        http://localhost:5000
"""

import asyncio, base64, io, json, os, re, sys, threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
_seq_model       = None
_seq_classes     = None
_seq_device      = None
_translator      = None
_tts_model       = None
_tts_tokenizer   = None
_parler_ok       = False
_pose_landmarker = None
_hand_landmarker = None
_predict_lock    = threading.Lock()   # MediaPipe is not thread-safe

# ── Fast TTS — mms-tts-kan ────────────────────────────────────────────────────
_fast_tts_model     = None
_fast_tts_tokenizer = None

# ── translation cache (Fix 3) ─────────────────────────────────────────────────
# Avoids hitting the 2-4s GPU model for words already translated this session.
# Key: English word (lowercase) → Kannada translation string
_translation_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

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
def sequence_model_path() -> Path:
    return ARTIFACTS / "sign_bilstm.pt"


def get_sequence_model():
    """New INCLUDE BiLSTM (263-class after full train). None until that file exists."""
    global _seq_model, _seq_classes, _seq_device
    path = sequence_model_path()
    if not path.exists():
        return None
    if _seq_model is None:
        from sequence_model import load_bundle
        _seq_model, _seq_classes, _seq_device = load_bundle(path)
    return _seq_model, _seq_classes, _seq_device


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
    global _tts_model, _tts_tokenizer, _parler_ok
    if _tts_model is None:
        import torch
        from transformers import VitsModel, AutoTokenizer as AT
        device = "cuda" if torch.cuda.is_available() else "cpu"

        parler_cached = Path.home().joinpath(
            ".cache/huggingface/hub/models--ai4bharat--indic-parler-tts-pretrained").exists()

        if parler_cached:
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                model_id = "ai4bharat/indic-parler-tts-pretrained"
                _tts_tokenizer = AT.from_pretrained(model_id)
                _tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device)
                _tts_model._tts_type = "parler"
                _parler_ok = True
                print("✓ Parler TTS loaded", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠ Parler TTS failed ({exc}) — using MMS Kannada instead", flush=True)
                _tts_model = None
                _tts_tokenizer = None
                _parler_ok = False

        if _tts_model is None:
            _tts_model = VitsModel.from_pretrained("facebook/mms-tts-kan").to(device)
            _tts_tokenizer = AT.from_pretrained("facebook/mms-tts-kan")
            _tts_model._tts_type = "mms"
            _parler_ok = False

        _tts_model.eval()
    return _tts_model, _tts_tokenizer


def get_fast_tts():
    global _fast_tts_model, _fast_tts_tokenizer
    if _fast_tts_model is None:
        import torch
        from transformers import VitsModel, AutoTokenizer as AT
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _fast_tts_model = VitsModel.from_pretrained("facebook/mms-tts-kan").to(device)
        _fast_tts_tokenizer = AT.from_pretrained("facebook/mms-tts-kan")
        _fast_tts_model.eval()
    return _fast_tts_model, _fast_tts_tokenizer


def _mms_wav(text: str) -> bytes:
    import torch, scipy.io.wavfile
    model, tokenizer = get_fast_tts()
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        audio = model(**inputs).waveform.squeeze().cpu().numpy().astype(np.float32)
    sample_rate = model.config.sampling_rate
    audio_i16 = (audio / max(np.abs(audio).max(), 1e-6) * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, audio_i16)
    return buf.getvalue()


def synthesize_wav(text: str, voice: str = DEFAULT_VOICE, fast: bool = False) -> bytes:
    """Generate Kannada speech.
    fast=True  → mms-tts-kan (<0.5s)
    fast=False → parler if available, else MMS
    """
    import torch, scipy.io.wavfile

    text = (text or "").strip()
    if not text:
        raise ValueError("empty TTS text")

    if fast:
        return _mms_wav(text)

    try:
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
            sample_rate = model.config.sampling_rate
            audio_i16 = (audio / max(np.abs(audio).max(), 1e-6) * 32767).astype(np.int16)
            buf = io.BytesIO()
            scipy.io.wavfile.write(buf, sample_rate, audio_i16)
            return buf.getvalue()

        return _mms_wav(text)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ TTS quality path failed ({exc}) — falling back to MMS", flush=True)
        return _mms_wav(text)

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
def _hand_frame_coverage(seq: np.ndarray) -> float:
    """Fraction of frames with at least one hand landmark present."""
    if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] < 225:
        return 0.0
    left = np.abs(seq[:, 99:162]).sum(axis=1) > 1e-4
    right = np.abs(seq[:, 162:225]).sum(axis=1) > 1e-4
    return float(np.mean(left | right))


def _pose_wrist_coverage(seq: np.ndarray) -> float:
    """MediaPipe Hands often misses on webcam; pose wrists (15/16) are a fallback."""
    if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] < 99:
        return 0.0
    # pose flattened: landmark i → indices 3i..3i+2
    lw = np.abs(seq[:, 15 * 3:15 * 3 + 3]).sum(axis=1) > 1e-4
    rw = np.abs(seq[:, 16 * 3:16 * 3 + 3]).sum(axis=1) > 1e-4
    return float(np.mean(lw | rw))


def _signing_activity(seq: np.ndarray) -> float:
    """Prefer hand landmarks; fall back to pose wrists so live signing still works."""
    hands = _hand_frame_coverage(seq)
    if hands >= 0.12:
        return hands
    return max(hands, _pose_wrist_coverage(seq) * 0.85)


def predict_frame_sequence(frames_bgr: list) -> list[tuple[str, float]]:
    import mediapipe as mp
    from extract_landmarks import frame_feature

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

    seq = np.stack(feats)
    activity = _signing_activity(seq)
    # Only block empty / black frames — do not require perfect hand detection.
    pose_energy = float(np.abs(seq[:, :99]).mean()) if seq.size else 0.0
    if activity < 0.08 and pose_energy < 1e-4:
        return [("__hands__", 0.0)]

    bundle = get_sequence_model()
    if bundle is not None:
        from sequence_model import predict_topk
        model, classes, device = bundle
        return predict_topk(model, classes, seq, device, k=5)

    from train_classifier import sequence_to_features
    x = sequence_to_features(seq).reshape(1, -1)
    clf, le = get_classifier()
    proba = clf.predict_proba(x)[0]
    idx = np.argsort(proba)[::-1][:5]
    return [(le.classes_[i], float(proba[i])) for i in idx]


def translate_text(text: str) -> str:
    """Translate English gloss to Kannada. Results are cached to avoid repeat GPU calls."""
    import torch
    key = text.strip().lower()

    # Check cache first (Fix 3)
    with _cache_lock:
        if key in _translation_cache:
            return _translation_cache[key]

    model, tokenizer, ip, device = get_translator()
    batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding="longest").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, num_beams=5,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    result = ip.postprocess_batch(decoded, lang="kan_Knda")[0]

    # Store in cache
    with _cache_lock:
        _translation_cache[key] = result

    return result


def translate_words_batch(words: list[str]) -> list[str]:
    """Translate a list of English words to Kannada efficiently.
    Uses cache for already-seen words, batches new ones in a single GPU call.
    Fix 2: sentence-level batch translation.
    """
    import torch
    results = [""] * len(words)
    to_translate = []   # (original_index, word)

    with _cache_lock:
        for i, word in enumerate(words):
            key = word.strip().lower()
            if key in _translation_cache:
                results[i] = _translation_cache[key]
            else:
                to_translate.append((i, word))

    if not to_translate:
        return results   # all cached

    model, tokenizer, ip, device = get_translator()
    batch_words = [w for _, w in to_translate]
    batch = ip.preprocess_batch(batch_words, src_lang="eng_Latn", tgt_lang="kan_Knda")
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding="longest").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, num_beams=5,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    translations = ip.postprocess_batch(decoded, lang="kan_Knda")

    with _cache_lock:
        for (i, word), translation in zip(to_translate, translations):
            key = word.strip().lower()
            _translation_cache[key] = translation
            results[i] = translation

    return results


# ── lifespan: preload fast models at startup ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    print("▶ Preloading classifier + MediaPipe...", flush=True)
    if sequence_model_path().exists():
        await loop.run_in_executor(None, get_sequence_model)
    elif (ARTIFACTS / "sign_classifier.joblib").exists():
        await loop.run_in_executor(None, get_classifier)
    await loop.run_in_executor(None, _init_mediapipe)
    print("▶ Preloading TTS...", flush=True)
    await loop.run_in_executor(None, get_fast_tts)   # mms — used for live Speak
    print("✓ Fast models ready — server is live", flush=True)
    yield
    # shutdown: nothing to clean up


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SilentTalk API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request/response models ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    frames: list[str]

class TranslateRequest(BaseModel):
    text: str

class TranslateBatchRequest(BaseModel):
    words: List[str]   # list of English words to translate in one call

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    fast: Optional[bool] = False   # True = mms (instant), False = parler (quality)


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    clf_ok  = sequence_model_path().exists() or (ARTIFACTS / "sign_classifier.joblib").exists()
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
    # Folder can exist but be corrupt — only claim Parler if it actually loaded.
    parler_ready = bool(_parler_ok)
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
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480.0 / w
            frame = cv2.resize(frame, (480, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        frames_bgr.append(frame)

    if not frames_bgr:
        raise HTTPException(400, "Could not decode frames")

    loop  = asyncio.get_event_loop()
    preds = await loop.run_in_executor(None, predict_frame_sequence, frames_bgr)

    if not preds:
        raise HTTPException(500, "No predictions returned")

    top_label, top_conf = preds[0]
    if top_label in ("__idle__", "__hands__") or top_conf <= 0:
        return {
            "top_label": "",
            "top_conf": 0.0,
            "top5": [],
            "idle": True,
            "reason": "Move into frame and sign — keep hands visible",
        }

    margin = 0.0
    if len(preds) >= 2:
        margin = float(top_conf - preds[1][1])
    # Soft hint only — do not hide the prediction from the UI
    uncertain = top_conf < 0.22

    return {
        "top_label": top_label,
        "top_conf":  round(top_conf * 100, 1),
        "top5": [{"label": l, "conf": round(c * 100, 1)} for l, c in preds],
        "margin": round(margin * 100, 1),
        "uncertain": uncertain,
    }


@app.post("/api/translate")
async def api_translate(req: TranslateRequest):
    if not req.text:
        raise HTTPException(400, "No text provided")
    loop        = asyncio.get_event_loop()
    translation = await loop.run_in_executor(None, translate_text, req.text)
    return {"translation": translation}


@app.post("/api/translate_batch")
async def api_translate_batch(req: TranslateBatchRequest):
    """Translate multiple words in one GPU call. Cache-aware.
    Fix 2: used by Speak Sentence to translate full sentence efficiently.
    """
    if not req.words:
        raise HTTPException(400, "No words provided")
    loop   = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, translate_words_batch, req.words)
    return {
        "translations": results,
        "pairs": [{"word": w, "translation": t} for w, t in zip(req.words, results)],
    }


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    if not req.text:
        raise HTTPException(400, "No text provided")
    loop = asyncio.get_event_loop()
    # fast=True → mms (<0.5s) for single word Speak button
    # fast=False → parler (3-6s, quality) for full Speak Sentence
    wav_bytes = await loop.run_in_executor(
        None, synthesize_wav, req.text, req.voice or DEFAULT_VOICE, req.fast or False)
    b64 = base64.b64encode(wav_bytes).decode()
    return {"audio_b64": b64, "format": "wav", "fast": req.fast}


@app.get("/sample/{folder}/{filename}")
async def serve_sample(folder: str, filename: str):
    path = SAMPLES_DIR / folder / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


# ── serve React SPA (built files; dev UI runs on Vite :5173) ──────────────────
REACT_DIR = ROOT / "static" / "react"
if REACT_DIR.exists():
    app.mount("/assets", StaticFiles(directory=REACT_DIR / "assets"), name="assets")

@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    """Serve built React. While coding UI, use Vite on port 5173."""
    index = REACT_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "error": "Frontend not built.",
            "hint": "Run: cd frontend && npm run dev  (open http://localhost:5173)",
        },
        404,
    )


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SilentTalk backend")
    parser.add_argument(
        "--https",
        action="store_true",
        help="HTTPS with certs/dev-*.pem so phone browsers can use the camera on LAN",
    )
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    ssl_kwargs = {}
    scheme = "http"
    if args.https:
        cert = ROOT / "certs" / "dev-cert.pem"
        key = ROOT / "certs" / "dev-key.pem"
        if not cert.exists() or not key.exists():
            print("Creating self-signed cert for phone camera access...")
            from isl_recognition.make_dev_cert import main as make_cert

            if make_cert() != 0:
                raise SystemExit(1)
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"

    print(f"▶ Backend  — {scheme}://localhost:{args.port}")
    if args.https:
        print(f"▶ Phone    — {scheme}://192.168.1.10:{args.port}  (accept certificate warning)")
    else:
        print("▶ Phone camera needs HTTPS — restart with:  python app.py --https")
    print("▶ UI dev   — cd frontend && npm run dev  →  http://localhost:5173")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
        workers=1,
        **ssl_kwargs,
    )
