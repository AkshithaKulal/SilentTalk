import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Camera, Square, Loader2, CameraOff, Zap, Radio } from "lucide-react"

const LIVE_INTERVAL_MS = 2500   // predict every 2.5 seconds
const CAPTURE_FRAMES  = 20      // collect 20 frames over 2 seconds

export default function WebcamCapture({ selectedSign, onPrediction }) {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)   // overlay canvas
  const liveTimerRef = useRef(null)
  const frameTimerRef = useRef(null)

  const [stream, setStream]         = useState(null)
  const [liveMode, setLiveMode]     = useState(false)
  const [processing, setProcessing] = useState(false)
  const [capturing, setCapturing]   = useState(false)
  const [livePred, setLivePred]     = useState(null)   // { label, conf }
  const [countdown, setCountdown]   = useState(0)
  const [progress, setProgress]     = useState(0)
  const [error, setError]           = useState("")

  // ── Draw overlay on canvas ──────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!livePred) return

    const { label, conf } = livePred
    const color = conf >= 70 ? "#22c55e" : conf >= 40 ? "#f59e0b" : "#ef4444"
    const w = canvas.width, h = canvas.height

    // Bottom bar background
    ctx.fillStyle = "rgba(0,0,0,0.65)"
    ctx.fillRect(0, h - 68, w, 68)

    // Confidence bar
    ctx.fillStyle = "rgba(255,255,255,0.1)"
    ctx.fillRect(12, h - 16, w - 24, 8)
    ctx.fillStyle = color
    ctx.fillRect(12, h - 16, (w - 24) * (conf / 100), 8)

    // Label text
    ctx.font = "bold 22px Inter, system-ui, sans-serif"
    ctx.fillStyle = "#ffffff"
    ctx.fillText(label, 14, h - 38)

    // Confidence text
    ctx.font = "13px Inter, system-ui, sans-serif"
    ctx.fillStyle = color
    ctx.fillText(`${conf}%`, 14, h - 22)

    // Live indicator dot
    ctx.beginPath()
    ctx.arc(w - 16, h - 52, 5, 0, Math.PI * 2)
    ctx.fillStyle = "#ef4444"
    ctx.fill()
    ctx.font = "11px Inter"
    ctx.fillStyle = "#ef4444"
    ctx.fillText("LIVE", w - 42, h - 47)
  }, [livePred])

  // ── Collect frames helper ───────────────────────────────────────────────────
  const collectFrames = useCallback(() => {
    return new Promise((resolve) => {
      const frames = []
      const cap = document.createElement("canvas")
      cap.width = 320; cap.height = 240
      const ctx = cap.getContext("2d")
      const TOTAL = CAPTURE_FRAMES
      let count = 0
      const interval = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, 320, 240)
          frames.push(cap.toDataURL("image/jpeg", 0.7))
        }
        count++
        if (count >= TOTAL) { clearInterval(interval); resolve(frames) }
      }, 2000 / TOTAL)   // spread over 2s
    })
  }, [])

  // ── Predict helper ──────────────────────────────────────────────────────────
  const runPredict = useCallback(async (frames) => {
    if (!frames.length) return
    try {
      const res = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames })
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setLivePred({ label: data.top_label, conf: data.top_conf })
      onPrediction(data)
    } catch (e) { setError("Prediction error: " + e.message) }
  }, [onPrediction])

  // ── Live mode loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!liveMode || !stream) return
    let active = true

    const loop = async () => {
      while (active) {
        setCapturing(true)
        const frames = await collectFrames()
        setCapturing(false)
        if (!active) break
        setProcessing(true)
        await runPredict(frames)
        setProcessing(false)
        if (!active) break
        // brief pause before next round
        await new Promise(r => setTimeout(r, 300))
      }
    }
    loop()
    return () => { active = false }
  }, [liveMode, stream, collectFrames, runPredict])

  // ── Manual capture ──────────────────────────────────────────────────────────
  const startManualCapture = async () => {
    if (!stream || capturing || processing) return
    setCapturing(true); setCountdown(3); setProgress(0)
    const frames = []
    const canvas = document.createElement("canvas")
    canvas.width = 320; canvas.height = 240
    const ctx = canvas.getContext("2d")
    const DURATION = 3000, INTERVAL = 100
    let elapsed = 0
    await new Promise(resolve => {
      const interval = setInterval(() => {
        elapsed += INTERVAL
        setProgress(elapsed / DURATION * 100)
        setCountdown(Math.ceil((DURATION - elapsed) / 1000))
        ctx.drawImage(videoRef.current, 0, 0, 320, 240)
        frames.push(canvas.toDataURL("image/jpeg", 0.7))
        if (elapsed >= DURATION) { clearInterval(interval); resolve() }
      }, INTERVAL)
    })
    setCapturing(false); setCountdown(0)
    setProcessing(true)
    await runPredict(frames)
    setProcessing(false); setProgress(0)
  }

  const startCamera = async () => {
    setError("")
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      setStream(s)
      if (videoRef.current) videoRef.current.srcObject = s
    } catch (e) { setError("Camera access denied: " + e.message) }
  }

  const stopCamera = () => {
    setLiveMode(false)
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
    setLivePred(null)
  }

  const toggleLive = () => {
    if (!stream) return
    setLiveMode(v => !v)
    if (liveMode) setLivePred(null)
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Camera card */}
      <div style={{ background: "#0d1117", border: `1px solid ${liveMode ? "rgba(239,68,68,0.3)" : "rgba(255,255,255,0.06)"}`,
        borderRadius: 16, overflow: "hidden",
        boxShadow: liveMode ? "0 0 20px rgba(239,68,68,0.1)" : "none",
        transition: "all 0.3s" }}>

        <div style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center",
          borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Webcam</span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {liveMode && (
              <motion.span animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1, repeat: Infinity }}
                style={{ fontSize: 11, color: "#ef4444", display: "flex", alignItems: "center", gap: 4, fontWeight: 700 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#ef4444",
                  boxShadow: "0 0 8px #ef4444" }} />
                LIVE PREDICTION
              </motion.span>
            )}
            {!liveMode && stream && (
              <span style={{ fontSize: 11, color: "#34d399", display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#34d399" }} />
                Camera On
              </span>
            )}
            {!stream && <span style={{ fontSize: 11, color: "#334155" }}>Off</span>}
          </div>
        </div>

        {/* Video + overlay */}
        <div style={{ position: "relative", background: "#060912", aspectRatio: "4/3" }}>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
          <canvas ref={canvasRef} width={640} height={480}
            style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} />

          {!stream && (
            <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 10 }}>
              <CameraOff size={36} color="#1e293b" />
              <p style={{ fontSize: 13, color: "#334155" }}>Camera not started</p>
            </div>
          )}

          {/* Corner guides */}
          {stream && !liveMode && !capturing && !processing && (() => {
            const positions = [
              { top: 10, left: 10, borderTop: "2px solid rgba(99,102,241,0.4)", borderLeft: "2px solid rgba(99,102,241,0.4)" },
              { top: 10, right: 10, borderTop: "2px solid rgba(99,102,241,0.4)", borderRight: "2px solid rgba(99,102,241,0.4)" },
              { bottom: 10, left: 10, borderBottom: "2px solid rgba(99,102,241,0.4)", borderLeft: "2px solid rgba(99,102,241,0.4)" },
              { bottom: 10, right: 10, borderBottom: "2px solid rgba(99,102,241,0.4)", borderRight: "2px solid rgba(99,102,241,0.4)" },
            ]
            return positions.map((pos, i) => (
              <div key={i} style={{ position: "absolute", width: 16, height: 16, ...pos }} />
            ))
          })()}

          {/* Manual capture countdown */}
          <AnimatePresence>
            {capturing && !liveMode && countdown > 0 && (
              <motion.div key={countdown} initial={{ scale: 1.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.4, opacity: 0 }}
                style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <div style={{ width: 80, height: 80, borderRadius: "50%", background: "rgba(0,0,0,0.75)",
                  border: "3px solid #ef4444", display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "0 0 30px rgba(239,68,68,0.4)" }}>
                  <span style={{ fontSize: 42, fontWeight: 900, color: "white" }}>{countdown}</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing overlay (manual mode only) */}
          <AnimatePresence>
            {processing && !liveMode && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ position: "absolute", inset: 0, background: "rgba(6,9,18,0.85)",
                  display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10 }}>
                <Loader2 size={28} color="#818cf8" style={{ animation: "spin 1s linear infinite" }} />
                <p style={{ fontSize: 13, color: "#818cf8", fontWeight: 600 }}>Analyzing...</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Live mode: subtle pulsing border when collecting */}
          {liveMode && capturing && (
            <div style={{ position: "absolute", inset: 0, border: "2px solid rgba(239,68,68,0.4)",
              borderRadius: 0, animation: "pulse 1s infinite", pointerEvents: "none" }} />
          )}

          {/* Progress bar */}
          {!liveMode && (capturing || processing) && (
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 3, background: "rgba(255,255,255,0.05)" }}>
              <motion.div animate={{ width: `${progress}%` }} transition={{ duration: 0.1 }}
                style={{ height: "100%", background: "linear-gradient(to right, #6366f1, #a78bfa)" }} />
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "grid", gridTemplateColumns: stream ? "1fr 1fr 1fr" : "1fr", gap: 8 }}>
        {!stream ? (
          <motion.button onClick={startCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              padding: "12px 0", borderRadius: 12, border: "none", cursor: "pointer",
              background: "linear-gradient(135deg, #4f46e5, #7c3aed)", color: "white",
              fontSize: 13, fontWeight: 700, boxShadow: "0 4px 20px rgba(99,102,241,0.3)" }}>
            <Camera size={15} /> Start Camera
          </motion.button>
        ) : (
          <>
            <motion.button onClick={stopCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                padding: "11px 0", borderRadius: 12, cursor: "pointer", fontSize: 13, fontWeight: 600,
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#64748b" }}>
              <Square size={13} /> Stop
            </motion.button>

            <motion.button onClick={toggleLive} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                padding: "11px 0", borderRadius: 12, border: "none", cursor: "pointer",
                background: liveMode ? "linear-gradient(135deg, #dc2626, #ef4444)" : "rgba(239,68,68,0.12)",
                border: `1px solid ${liveMode ? "transparent" : "rgba(239,68,68,0.25)"}`,
                color: liveMode ? "white" : "#f87171",
                fontSize: 13, fontWeight: 700,
                boxShadow: liveMode ? "0 4px 16px rgba(239,68,68,0.3)" : "none" }}>
              <Radio size={13} />
              {liveMode ? "Stop Live" : "Go Live"}
            </motion.button>

            <motion.button onClick={startManualCapture} disabled={capturing || processing || liveMode}
              whileHover={!capturing && !processing && !liveMode ? { scale: 1.02 } : {}}
              whileTap={!capturing && !processing && !liveMode ? { scale: 0.97 } : {}}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                padding: "11px 0", borderRadius: 12, border: "none",
                cursor: capturing || processing || liveMode ? "not-allowed" : "pointer",
                background: capturing || processing || liveMode ? "rgba(34,197,94,0.1)" : "linear-gradient(135deg, #16a34a, #22c55e)",
                border: `1px solid ${capturing || processing || liveMode ? "rgba(34,197,94,0.2)" : "transparent"}`,
                color: capturing || processing || liveMode ? "#4ade80" : "white",
                fontSize: 13, fontWeight: 700, opacity: liveMode ? 0.4 : 1,
                boxShadow: capturing || processing || liveMode ? "none" : "0 4px 16px rgba(34,197,94,0.25)" }}>
              {processing && !liveMode ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={13} />}
              {capturing && !liveMode ? `${countdown}s` : "Capture"}
            </motion.button>
          </>
        )}
      </div>

      {error && (
        <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "#fca5a5" }}>
          {error}
        </div>
      )}

      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)",
        borderRadius: 10, padding: "10px 14px" }}>
        <p style={{ fontSize: 11, color: "#475569", lineHeight: 1.6 }}>
          <span style={{ color: "#94a3b8", fontWeight: 600 }}>Go Live</span> — auto-predicts every 2.5s with label overlaid on feed |
          <span style={{ color: "#94a3b8", fontWeight: 600 }}> Capture</span> — manual 3s recording
        </p>
      </div>
    </div>
  )
}
