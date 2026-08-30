#!/usr/bin/env python3
"""Real-time ISL Recognition GUI with live webcam prediction and TTS."""

import sys
import os
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

# Force UTF-8 encoding for console output (handles Kannada/Indic scripts on Windows)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import cv2
    from PIL import Image, ImageTk
    import numpy as np
    import joblib
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install: pip install opencv-python pillow torch transformers IndicTransToolkit")
    sys.exit(1)

from predict_sign import predict_video_topk


class IndicTranslator:
    """Wrapper for IndicTrans2 English→Indic translation."""

    def __init__(self, target_lang: str = "kan_Knda", device: str | None = None):
        self.target_lang = target_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "ai4bharat/indictrans2-en-indic-dist-200M"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="eager"
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.processor = IndicProcessor(inference=True)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            batch = self.processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=self.target_lang)
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translation = self.processor.postprocess_batch(decoded, lang=self.target_lang)[0]
            return translation
        except Exception as e:
            return ""


def speak_sentence(sentence: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(sentence)
        engine.runAndWait()
        return True
    except Exception:
        return False


class SilentTalkGUI:
    """Real-time ISL Recognition GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("SilentTalk - Real-Time ISL Recognition")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        # State
        self.cap = None
        self.running = False
        self.translator = None
        self.prediction_history = deque(maxlen=10)
        self.confidence_threshold = 0.5
        self.target_lang = "kan_Knda"
        self.enable_tts = tk.BooleanVar(value=True)
        self.enable_translation = tk.BooleanVar(value=True)

        # Paths
        self.artifacts_path = Path(__file__).parent / "transfer_pack"
        self.models_path = Path(__file__).parent / "models"

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build GUI layout."""
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=10, pady=10)

        title = ttk.Label(header, text="🎬 SilentTalk - Real-Time ISL Recognition", font=("Arial", 18, "bold"))
        title.pack(side=tk.LEFT)

        status_label = ttk.Label(header, text="Status: Ready", font=("Arial", 12))
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label

        # Main content
        content = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Video feed
        left_panel = ttk.Frame(content)
        content.add(left_panel, weight=2)

        video_label = ttk.Label(left_panel, text="Live Webcam Feed", font=("Arial", 12, "bold"))
        video_label.pack(pady=5)

        self.video_canvas = tk.Canvas(left_panel, width=640, height=480, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Video controls
        video_controls = ttk.Frame(left_panel)
        video_controls.pack(fill=tk.X, pady=10)

        self.start_button = ttk.Button(video_controls, text="▶ Start", command=self.start_webcam)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(video_controls, text="⏹ Stop", command=self.stop_webcam, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(video_controls, text="📸 Snapshot", command=self.take_snapshot).pack(side=tk.LEFT, padx=5)

        # Right panel: Predictions
        right_panel = ttk.Frame(content)
        content.add(right_panel, weight=1)

        # Current prediction
        pred_label = ttk.Label(right_panel, text="Current Prediction", font=("Arial", 12, "bold"))
        pred_label.pack(pady=5)

        self.pred_frame = ttk.Frame(right_panel, relief=tk.SUNKEN, borderwidth=2)
        self.pred_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.sign_label = ttk.Label(self.pred_frame, text="--", font=("Arial", 14, "bold"), foreground="white", background="#2d2d2d")
        self.sign_label.pack(pady=10, fill=tk.X)

        self.confidence_label = ttk.Label(self.pred_frame, text="Confidence: --", font=("Arial", 10))
        self.confidence_label.pack(pady=5)

        self.translated_label = ttk.Label(self.pred_frame, text="Translated: --", font=("Arial", 10), wraplength=200)
        self.translated_label.pack(pady=5, fill=tk.X)

        # History
        hist_label = ttk.Label(right_panel, text="Prediction History", font=("Arial", 12, "bold"))
        hist_label.pack(pady=5)

        self.history_listbox = tk.Listbox(right_panel, height=8, font=("Courier", 9))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)

        # Settings
        settings_label = ttk.Label(right_panel, text="Settings", font=("Arial", 12, "bold"))
        settings_label.pack(pady=5, fill=tk.X)

        settings_frame = ttk.LabelFrame(right_panel, text="")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Threshold slider
        ttk.Label(settings_frame, text="Confidence Threshold").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.threshold_var)
        threshold_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.threshold_label = ttk.Label(settings_frame, text="0.50")
        self.threshold_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        self.threshold_var.trace("w", lambda *args: self._update_threshold_label())

        # Language selector
        ttk.Label(settings_frame, text="Language").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.lang_var = tk.StringVar(value="kan_Knda")
        lang_menu = ttk.Combobox(settings_frame, textvariable=self.lang_var, values=["kan_Knda", "tul_Knda"], state="readonly")
        lang_menu.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)

        # Checkboxes
        ttk.Checkbutton(settings_frame, text="Enable TTS", variable=self.enable_tts).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(settings_frame, text="Enable Translation", variable=self.enable_translation).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        settings_frame.columnconfigure(1, weight=1)

    def _update_threshold_label(self):
        val = self.threshold_var.get()
        self.threshold_label.config(text=f"{val:.2f}")
        self.confidence_threshold = val

    def start_webcam(self):
        """Start webcam capture."""
        if not self.translator:
            try:
                self.status_label.config(text="Loading IndicTrans2 model...")
                self.root.update()
                self.translator = IndicTranslator(target_lang=self.target_lang)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load translator: {e}")
                return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running")

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_webcam(self):
        """Stop webcam capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

    def _capture_loop(self):
        """Capture and process frames."""
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            # Convert to PIL and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)

            self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.video_canvas.image = photo  # Keep reference

            frame_count += 1
            if frame_count % 15 == 0:  # Process every 15 frames (~0.5s at 30fps)
                self._predict_frame(frame)

            self.root.update_idletasks()

    def _predict_frame(self, frame):
        """Predict sign from frame."""
        # Save frame temporarily
        temp_path = Path(self.models_path).parent / "temp_frame.jpg"
        cv2.imwrite(str(temp_path), frame)

        try:
            # This would need a webcam-specific prediction function
            # For now, just display placeholder
            self.sign_label.config(text="Detecting...", foreground="yellow")
        except Exception as e:
            self.sign_label.config(text=f"Error: {str(e)[:30]}", foreground="red")

    def take_snapshot(self):
        """Save current frame."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(Path(self.artifacts_path).parent / filename), frame)
                messagebox.showinfo("Success", f"Saved: {filename}")

    def update_prediction(self, label: str, confidence: float, translated: str = ""):
        """Update prediction display."""
        self.sign_label.config(text=label, foreground="lime" if confidence >= self.confidence_threshold else "orange")
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        self.translated_label.config(text=f"Translated: {translated}")

        # Add to history
        entry = f"{datetime.now().strftime('%H:%M:%S')} | {label} ({confidence*100:.1f}%)"
        self.history_listbox.insert(0, entry)

        self.prediction_history.append((label, confidence, translated))

    def run(self):
        """Run GUI."""
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = SilentTalkGUI(root)
    app.run()
