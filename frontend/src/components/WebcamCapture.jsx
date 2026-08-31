import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Camera, Square, Loader2, Zap, Radio } from "lucide-react"

const LIVE_CAPTURE_FRAMES   = 18
const MANUAL_CAPTURE_FRAMES = 24
const LIVE_COLLECT_MS       = 1600
const LIVE_GAP_MS           = 280
const FRAME_W               = 480
const FRAME_H               = 360

export default function WebcamCapture({ selectedSign, onPrediction, isSpeaking }) {
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

  // ── Canvas: signing guide frame only (no bottom HUD over hands) ───────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!stream) return

    const w = canvas.width, h = canvas.height
    const padX = w * 0.08
    const padTop = h * 0.06
    const padBottom = h * 0.22
    const gw = w - padX * 2
    const gh = h - padTop - padBottom

    ctx.strokeStyle = liveMode ? "rgba(225, 29, 72, 0.55)" : "rgba(255, 255, 255, 0.35)"
    ctx.lineWidth = 2
    ctx.setLineDash([10, 8])
    ctx.strokeRect(padX, padTop, gw, gh)
    ctx.setLineDash([])

    ctx.font = "600 11px Inter, system-ui, sans-serif"
    ctx.fillStyle = "rgba(255,255,255,0.45)"
    ctx.fillText("Guide only — sign naturally", padX + 8, padTop + 16)
  }, [stream, liveMode])

  // Keep video element attached to the MediaStream (fixes black / dead camera)
  useEffect(() => {
    const video = videoRef.current
    if (!video || !stream) return
    if (video.srcObject !== stream) {
      video.srcObject = stream
    }
    const play = async () => {
      try {
        await video.play()
      } catch (e) {
        setError("Camera started but video could not play: " + (e?.message || e))
      }
    }
    play()
  }, [stream])

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
      setLivePred({ label: data.top_label, conf: data.top_conf, idle: !!data.idle, reason: data.reason || "" })
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
    const insecureLan =
      typeof window !== "undefined" &&
      !window.isSecureContext &&
      !/^localhost$|^127\.0\.0\.1$/.test(window.location.hostname)

    if (insecureLan || !navigator.mediaDevices?.getUserMedia) {
      setError(
        "Phone camera needs HTTPS. On the laptop run: python app.py --https  then open https://192.168.1.10:5000 and tap Advanced → Proceed."
      )
      return
    }
    // Close any previous stream first
    stream?.getTracks().forEach((t) => t.stop())

    const attempts = [
      { video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
      { video: { facingMode: "environment" }, audio: false },
      { video: true, audio: false },
    ]

    let lastErr = null
    for (const constraints of attempts) {
      try {
        const s = await navigator.mediaDevices.getUserMedia(constraints)
        setStream(s)
        setLiveMode(true)
        return
      } catch (e) {
        lastErr = e
      }
    }

    const name = lastErr?.name || "Error"
    const msg = lastErr?.message || String(lastErr)
    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      setError("Camera blocked — allow camera permission for this site, then Start live again.")
    } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      setError("No camera found on this device.")
    } else if (name === "NotReadableError" || name === "TrackStartError") {
      setError("Camera is busy — close other apps using the camera, then try again.")
    } else {
      setError(`Camera failed (${name}): ${msg}`)
    }
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


  return (
    <div className="cam-col">
      <div className={`cam-shell${liveMode ? " cam-shell--live" : ""}`}>
        <div className="cam-toolbar">
          <span className="cam-toolbar-title">
            {liveMode && <span className="live-pip" />}
            Camera{selectedSign ? ` · ${selectedSign.label}` : ""}
          </span>
          <div className="cam-toolbar-right">
            {stream && (
              <>
                <button type="button" className="cam-toolbar-btn" onClick={stopCamera}>
                  <Square size={12} /> Stop
                </button>
                <button
                  type="button"
                  className={`cam-toolbar-btn${liveMode ? " cam-toolbar-btn--live" : ""}`}
                  onClick={toggleLive}
                >
                  <Radio size={12} />
                  {liveMode ? "Listening" : "Go live"}
                </button>
              </>
            )}
            {statusBadge()}
          </div>
        </div>

        <div className="cam-viewport">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="cam-video"
          />
          <canvas ref={canvasRef} width={1280} height={720} className="cam-guide-canvas" />

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
                key={(livePred.label || "idle") + String(livePred.conf)}
                className="cam-pred-badge"
                initial={{ y: -8, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div>
                  <div className="cam-pred-label">
                    {livePred.idle ? "Show hands" : (livePred.label || "…")}
                  </div>
                  <div
                    className="cam-pred-conf"
                    style={{
                      color: livePred.idle
                        ? "#fbbf24"
                        : livePred.conf >= 72 ? "var(--ok)" : livePred.conf >= 40 ? "var(--warn)" : "var(--bad)",
                    }}
                  >
                    {livePred.idle
                      ? (livePred.reason || "Sign in front of the camera")
                      : isSpeaking
                        ? "Speaking Kannada…"
                        : `${livePred.conf}% · ${livePred.conf >= 58 ? "detected" : "hold sign"}`}
                  </div>
                </div>
                {liveMode && !livePred.idle && (
                  <span className="cam-pred-live">{isSpeaking ? "SPEAK" : "LIVE"}</span>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {stream && !liveMode && (
        <button
          type="button"
          onClick={startManualCapture}
          disabled={capturing || processing}
          className="cam-once-link"
        >
          {processing ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={12} />}
          {capturing ? `${countdown}s` : "Capture once instead"}
        </button>
      )}

      {error && (
        <div className="cam-error">{error}</div>
      )}
    </div>
  )
}
