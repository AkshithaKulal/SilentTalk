/* SilentTalk Frontend Logic */

let webcamStream = null;
let capturedFrames = [];
let currentTranslation = "";
let history = [];

// ── Status check ────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    document.getElementById("dot-clf").className   = "status-dot " + (data.classifier ? "ok" : "err");
    document.getElementById("dot-trans").className = "status-dot " + (data.translation_model && data.lora_adapter ? "ok" : "err");
    document.getElementById("dot-tts").className   = "status-dot " + (data.tts_model ? "ok" : "err");
  } catch { /* server not up yet */ }
}

// ── Webcam ───────────────────────────────────────────────────────────────────
async function startWebcam() {
  const video = document.getElementById("webcam");
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = webcamStream;
    document.getElementById("btn-start-webcam").textContent = "✓ Camera On";
    document.getElementById("btn-start-webcam").disabled = true;
    document.getElementById("btn-capture").disabled = false;
  } catch (e) {
    alert("Could not access webcam: " + e.message);
  }
}

// ── Sign selector ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkStatus();
  setInterval(checkStatus, 10000);

  document.querySelectorAll(".sign-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sign-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      loadSample(btn.dataset.folder, btn.dataset.sample, btn.dataset.label);
    });
  });
});

function loadSample(folder, file, label) {
  const section = document.getElementById("sample-section");
  const video = document.getElementById("sample-video");
  const lbl = document.getElementById("sample-label");

  lbl.textContent = label;
  video.src = `/sample/${folder}/${file}`;
  video.load();
  video.play();
  section.style.display = "block";
}

// ── Capture & predict ────────────────────────────────────────────────────────
function startCapture() {
  const video = document.getElementById("webcam");
  const badge = document.getElementById("recording-badge");
  const progressDiv = document.getElementById("capture-progress");
  const fill = document.getElementById("progress-fill");
  const statusTxt = document.getElementById("capture-status");
  const btn = document.getElementById("btn-capture");

  if (!webcamStream) { alert("Start camera first."); return; }

  btn.disabled = true;
  badge.style.display = "block";
  progressDiv.style.display = "block";
  capturedFrames = [];

  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 240;
  const ctx = canvas.getContext("2d");

  const DURATION_MS = 3000;
  const INTERVAL_MS = 100; // ~10fps capture
  const start = Date.now();

  statusTxt.textContent = "Capturing (3s)...";

  const interval = setInterval(() => {
    const elapsed = Date.now() - start;
    const pct = Math.min(100, (elapsed / DURATION_MS) * 100);
    fill.style.width = pct + "%";

    ctx.drawImage(video, 0, 0, 320, 240);
    capturedFrames.push(canvas.toDataURL("image/jpeg", 0.7));

    if (elapsed >= DURATION_MS) {
      clearInterval(interval);
      badge.style.display = "none";
      statusTxt.textContent = "Processing...";
      fill.style.width = "100%";
      runPrediction();
    }
  }, INTERVAL_MS);
}

async function runPrediction() {
  const btn = document.getElementById("btn-capture");
  const progressDiv = document.getElementById("capture-progress");
  const statusTxt = document.getElementById("capture-status");

  try {
    statusTxt.textContent = "Running classifier...";
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: capturedFrames })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    showPrediction(data);

    // Auto-translate
    statusTxt.textContent = "Translating...";
    const tRes = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: data.top_label })
    });
    const tData = await tRes.json();
    if (!tData.error) {
      showTranslation(tData.translation);
      currentTranslation = tData.translation;
      document.getElementById("btn-speak").disabled = false;
    }

    addHistory(data.top_label, data.top_conf, tData.translation || "");
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    progressDiv.style.display = "none";
  }
}

// ── Display helpers ──────────────────────────────────────────────────────────
function showPrediction(data) {
  const box = document.getElementById("result-box");
  const conf = data.top_conf;
  const confClass = conf >= 70 ? "" : conf >= 40 ? "med" : "low";
  box.innerHTML = `
    <div>
      <div class="result-label">${data.top_label}</div>
      <div class="result-conf ${confClass}">${conf}% confidence</div>
    </div>`;

  // Top-5
  const top5Box = document.getElementById("top5-box");
  const top5List = document.getElementById("top5-list");
  top5List.innerHTML = data.top5.map((p, i) => `
    <div class="pred-row">
      <span class="pred-rank">${i + 1}</span>
      <span class="pred-name">${p.label}</span>
      <div class="pred-bar-wrap"><div class="pred-bar" style="width:${p.conf}%"></div></div>
      <span class="pred-pct">${p.conf}%</span>
    </div>`).join("");
  top5Box.style.display = "block";
}

function showTranslation(text) {
  const box = document.getElementById("translation-box");
  document.getElementById("translation-text").textContent = text;
  box.style.display = "block";
}

function showError(msg) {
  const box = document.getElementById("result-box");
  box.innerHTML = `<div style="color:#ef4444;font-size:13px;">Error: ${msg}</div>`;
}

function addHistory(label, conf, translation) {
  const now = new Date().toLocaleTimeString();
  history.unshift({ label, conf, translation, time: now });
  const list = document.getElementById("history-list");
  list.innerHTML = history.slice(0, 15).map(h => `
    <div class="history-entry">
      <span><span class="hw">${h.label}</span> ${h.translation ? "→ " + h.translation : ""}</span>
      <span>${h.conf}% · ${h.time}</span>
    </div>`).join("");
}

// ── TTS ──────────────────────────────────────────────────────────────────────
async function speakTranslation() {
  if (!currentTranslation) return;
  const btn = document.getElementById("btn-speak");
  btn.disabled = true;
  btn.textContent = "⏳ Synthesizing...";

  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: currentTranslation })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Play audio in browser
    const byteChars = atob(data.audio_b64);
    const byteArr = new Uint8Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) byteArr[i] = byteChars.charCodeAt(i);
    const blob = new Blob([byteArr], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play();
    audio.onended = () => URL.revokeObjectURL(url);
  } catch (e) {
    alert("TTS error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔊 Speak";
  }
}
