import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Camera, Square, Loader2, Zap, Radio } from "lucide-react"

const LIVE_CAPTURE_FRAMES   = 12
const MANUAL_CAPTURE_FRAMES = 18
const LIVE_COLLECT_MS       = 1100
const LIVE_GAP_MS           = 200
const FRAME_W               = 320
const FRAME_H               = 240

export default function WebcamCapture({ selectedSign, onPrediction }) {
  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const onPredictionRef = useRef(onPrediction)
  useEffect(() => { onPredictionRef.current = onPrediction }, [onPrediction])

  const [stream, setStream]         = useState(null)
  const [liveMode, setLiveMode]     = useState(false)
  const [processing, setProcessing] = useState(false)
  const [capturing, setCapturing]   = useState(false)
  const [livePred, setLivePred]     = useState(null)
  const [countdown, setCountdown]   = useState(0)
  const [progress, setProgress]     = useState(0)
  const [nextIn, setNextIn]         = useState(0)
  const [error, setError]           = useState("")

  // ── Canvas overlay ──────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!livePred) return

    const { label, conf } = livePred
    const color = conf >= 70 ? "#047857" : conf >= 40 ? "#b45309" : "#b91c1c"
    const w = canvas.width, h = canvas.height

    // Bottom pill background
    const pillH = 60
    ctx.fillStyle = "rgba(255,255,255,0.92)"
    const r = 0
    ctx.fillRect(0, h - pillH, w, pillH)

    // Left accent bar
    ctx.fillStyle = color
    ctx.fillRect(0, h - pillH, 4, pillH)

    // Confidence fill bar
    ctx.fillStyle = "#e2e8f0"
    ctx.fillRect(12, h - 14, w - 24, 6)
    ctx.fillStyle = color
    ctx.fillRect(12, h - 14, (w - 24) * (conf / 100), 6)

    // Label
    ctx.font = "bold 20px Inter, system-ui, sans-serif"
    ctx.fillStyle = "#0f172a"
    ctx.fillText(label, 14, h - 32)

    // Conf text
    ctx.font = "600 12px Inter, system-ui, sans-serif"
    ctx.fillStyle = color
    ctx.fillText(`${conf}%`, 14, h - 18)

    // Live badge
    ctx.fillStyle = "#dc2626"
    ctx.beginPath()
    ctx.arc(w - 14, h - 46, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.font = "700 10px Inter"
    ctx.fillStyle = "#dc2626"
    ctx.fillText("LIVE", w - 40, h - 41)
  }, [livePred])

  // ── collectFrames ─────────────────────────────────────────────────────────
  const collectFrames = useCallback((frameCount = MANUAL_CAPTURE_FRAMES, collectMs = LIVE_COLLECT_MS) => {
    return new Promise((resolve) => {
      const frames = []
      const cap = document.createElement("canvas")
      cap.width = FRAME_W; cap.height = FRAME_H
      const ctx = cap.getContext("2d")
      let count = 0, done = false

      const finish = () => {
        if (done) return; done = true
        clearInterval(intervalId); clearTimeout(timeoutId); resolve(frames)
      }
      const intervalId = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, FRAME_W, FRAME_H)
          frames.push(cap.toDataURL("image/jpeg", 0.82))
          count++
        }
        if (count >= frameCount) finish()
      }, collectMs / frameCount)
      const timeoutId = setTimeout(finish, collectMs + 1000)
    })
  }, [])

  // ── runPredict ────────────────────────────────────────────────────────────
  const runPredict = useCallback(async (frames) => {
    if (!frames.length) {
      setError("No frames captured — ensure camera is visible and try again.")
      return
    }
    try {
      const res = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames })
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setError("")
      setLivePred({ label: data.top_label, conf: data.top_conf })
      onPredictionRef.current(data)
    } catch (e) { setError("Prediction error: " + e.message) }
  }, [])

  // ── Live loop ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!liveMode || !stream) return
    let active = true
    const loop = async () => {
      while (active) {
        setNextIn(0); setCapturing(true)
        const frames = await collectFrames(LIVE_CAPTURE_FRAMES, LIVE_COLLECT_MS)
        setCapturing(false)
        if (!active) break
        setProcessing(true)
        await runPredict(frames)
        setProcessing(false)
        if (!active) break
        await new Promise(r => setTimeout(r, LIVE_GAP_MS))
        setNextIn(0)
      }
    }
    loop()
    return () => { active = false; setNextIn(0) }
  }, [liveMode, stream])

  // ── Manual capture ────────────────────────────────────────────────────────
  const startManualCapture = async () => {
    if (!stream || capturing || processing) return
    setCapturing(true); setCountdown(3); setProgress(0)
    const frames = []
    const canvas = document.createElement("canvas")
    canvas.width = FRAME_W; canvas.height = FRAME_H
    const ctx = canvas.getContext("2d")
    const DURATION = 2200, INTERVAL = 110
    let elapsed = 0
    await new Promise(resolve => {
      const interval = setInterval(() => {
        elapsed += INTERVAL
        setProgress(elapsed / DURATION * 100)
        setCountdown(Math.ceil((DURATION - elapsed) / 1000))
        if (videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, FRAME_W, FRAME_H)
          frames.push(canvas.toDataURL("image/jpeg", 0.82))
        }
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
      setLiveMode(true)
    } catch (e) { setError("Camera access denied: " + e.message) }
  }

  const stopCamera = () => {
    setLiveMode(false); stream?.getTracks().forEach(t => t.stop())
    setStream(null); setLivePred(null); setNextIn(0)
  }

  const toggleLive = () => {
    if (!stream) return
    setLiveMode(v => !v)
    if (liveMode) { setLivePred(null); setNextIn(0) }
  }

  // ── Status badge content ───────────────────────────────────────────────────
  const statusBadge = () => {
    if (!stream) return <span style={{ fontSize: 12, color: "var(--faint)" }}>Camera off</span>
    const label = liveMode && capturing ? "Capturing"
      : liveMode && processing ? "Analyzing"
      : liveMode && nextIn > 0 ? `Next in ${nextIn}s`
      : liveMode ? "Live"
      : "Camera on"
    const color = liveMode ? "var(--live)" : "var(--ok)"
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 600, color }}>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: color }} />
        {label}
      </span>
    )
  }

  const btnBase = {
    height: 44,
    borderRadius: 10,
    fontSize: 14,
    fontWeight: 600,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    cursor: "pointer",
  }

  return (
    <div className="cam-col" style={{ gap: 10 }}>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          background: "var(--surface)",
          borderRadius: 22,
          overflow: "hidden",
          border: `1px solid ${liveMode ? "rgba(225,29,72,0.35)" : "var(--line)"}`,
          boxShadow: liveMode ? "0 0 0 4px rgba(225,29,72,0.08), var(--shadow)" : "var(--shadow)",
          transition: "border-color 0.25s ease, box-shadow 0.25s ease",
        }}
      >
        <div
          style={{
            padding: "10px 14px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderBottom: "1px solid var(--line)",
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
            {liveMode && <span className="live-pip" />}
            Camera{selectedSign ? ` · ${selectedSign.label}` : ""}
          </span>
          {statusBadge()}
        </div>

        <div style={{ position: "relative", flex: 1, minHeight: 0, background: "#0b1220" }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          />

          {!stream && (
            <div className="stage-empty">
              <span className="orb orb-a" />
              <span className="orb orb-b" />
              <span className="orb orb-c" />
              <div className="ring-wrap">
                <Camera size={28} color="#fff" />
              </div>
              <p style={{ fontSize: 18, fontWeight: 800, color: "#fff", zIndex: 1, letterSpacing: "-0.03em" }}>
                Your signs, out loud
              </p>
              <p style={{ fontSize: 13, color: "rgba(255,255,255,0.65)", zIndex: 1, maxWidth: 280, textAlign: "center" }}>
                Camera stays on this device. Press start and sign in frame.
              </p>
              <motion.button
                type="button"
                className="start-live-btn"
                onClick={startCamera}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <Camera size={16} /> Start live
              </motion.button>
            </div>
          )}

          {stream && !liveMode && !capturing && !processing &&
            [
              { top: 12, left: 12, borderTop: "2px solid rgba(255,255,255,0.55)", borderLeft: "2px solid rgba(255,255,255,0.55)" },
              { top: 12, right: 12, borderTop: "2px solid rgba(255,255,255,0.55)", borderRight: "2px solid rgba(255,255,255,0.55)" },
              { bottom: 12, left: 12, borderBottom: "2px solid rgba(255,255,255,0.55)", borderLeft: "2px solid rgba(255,255,255,0.55)" },
              { bottom: 12, right: 12, borderBottom: "2px solid rgba(255,255,255,0.55)", borderRight: "2px solid rgba(255,255,255,0.55)" },
            ].map((pos, i) => <div key={i} style={{ position: "absolute", width: 18, height: 18, ...pos }} />)}

          <AnimatePresence>
            {capturing && !liveMode && countdown > 0 && (
              <motion.div
                key={countdown}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}
              >
                <div
                  style={{
                    width: 72,
                    height: 72,
                    borderRadius: "50%",
                    background: "#fff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 32,
                    fontWeight: 800,
                    color: "var(--ink)",
                  }}
                >
                  {countdown}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {processing && !liveMode && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(17,24,39,0.55)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 10,
                }}
              >
                <Loader2 size={28} color="#fff" style={{ animation: "spin 1s linear infinite" }} />
                <p style={{ fontSize: 14, fontWeight: 600, color: "#fff" }}>Reading the sign</p>
              </motion.div>
            )}
          </AnimatePresence>

          {!liveMode && (capturing || processing) && (
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 3, background: "rgba(255,255,255,0.15)" }}>
              <div style={{ width: `${progress}%`, height: "100%", background: "var(--accent)" }} />
            </div>
          )}

          <AnimatePresence>
            {livePred && (
              <motion.div
                key={livePred.label}
                initial={{ y: 16, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  position: "absolute",
                  left: 12,
                  right: 12,
                  bottom: 12,
                  padding: "10px 14px",
                  borderRadius: 14,
                  background: "rgba(255,255,255,0.92)",
                  backdropFilter: "blur(10px)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 10,
                }}
              >
                <div>
                  <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.03em" }}>{livePred.label}</div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: livePred.conf >= 70 ? "var(--ok)" : livePred.conf >= 40 ? "var(--warn)" : "var(--bad)" }}>
                    {livePred.conf}% · {livePred.conf >= 60 ? "joining message" : "hold the sign"}
                  </div>
                </div>
                {liveMode && <span style={{ fontSize: 11, fontWeight: 800, color: "var(--live)", letterSpacing: "0.06em" }}>LIVE</span>}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {stream && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <button
            type="button"
            onClick={stopCamera}
            style={{ ...btnBase, background: "var(--surface)", border: "1px solid var(--line)", color: "var(--muted)" }}
          >
            <Square size={13} /> Stop
          </button>
          <button
            type="button"
            onClick={toggleLive}
            style={{
              ...btnBase,
              border: liveMode ? "none" : "1px solid var(--live)",
              background: liveMode ? "var(--live)" : "var(--surface)",
              color: liveMode ? "#fff" : "var(--live)",
              boxShadow: liveMode ? "0 8px 20px rgba(225,29,72,0.25)" : "none",
            }}
          >
            <Radio size={13} />
            {liveMode ? "Listening…" : "Go live"}
          </button>
        </div>
      )}

      {stream && !liveMode && (
        <button
          type="button"
          onClick={startManualCapture}
          disabled={capturing || processing}
          style={{
            alignSelf: "flex-start",
            background: "none",
            border: "none",
            color: "var(--muted)",
            fontSize: 12,
            fontWeight: 600,
            cursor: capturing || processing ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: 0,
          }}
        >
          {processing ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={12} />}
          {capturing ? `${countdown}s` : "Capture once instead"}
        </button>
      )}

      {error && (
        <div style={{ background: "var(--bad-soft)", border: "1px solid #fecaca", borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "var(--bad)" }}>
          {error}
        </div>
      )}

      {stream && (
        <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
          Keep signing. Confident words join the message. Pick a voice, then Speak.
        </p>
      )}
    </div>
  )
}
