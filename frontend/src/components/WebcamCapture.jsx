import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Camera, Square, Loader2, CameraOff, Zap, Radio } from "lucide-react"

const LIVE_CAPTURE_FRAMES   = 10
const MANUAL_CAPTURE_FRAMES = 20
const LIVE_COLLECT_MS       = 1500

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
    const color = conf >= 70 ? "#059669" : conf >= 40 ? "#d97706" : "#dc2626"
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
  const collectFrames = useCallback((frameCount = MANUAL_CAPTURE_FRAMES, collectMs = 2000) => {
    return new Promise((resolve) => {
      const frames = []
      const cap = document.createElement("canvas")
      cap.width = 160; cap.height = 120
      const ctx = cap.getContext("2d")
      let count = 0, done = false

      const finish = () => {
        if (done) return; done = true
        clearInterval(intervalId); clearTimeout(timeoutId); resolve(frames)
      }
      const intervalId = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, 160, 120)
          frames.push(cap.toDataURL("image/jpeg", 0.8))
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
        await new Promise(r => setTimeout(r, 500))
        if (!active) break
        const WAIT = 1500, steps = 6
        for (let i = steps; i >= 1; i--) {
          if (!active) break
          setNextIn(Math.round((i / steps) * (WAIT / 1000) * 10) / 10)
          await new Promise(r => setTimeout(r, WAIT / steps))
        }
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
    canvas.width = 160; canvas.height = 120
    const ctx = canvas.getContext("2d")
    const DURATION = 3000, INTERVAL = 100
    let elapsed = 0
    await new Promise(resolve => {
      const interval = setInterval(() => {
        elapsed += INTERVAL
        setProgress(elapsed / DURATION * 100)
        setCountdown(Math.ceil((DURATION - elapsed) / 1000))
        if (videoRef.current && videoRef.current.readyState >= 2) {
          ctx.drawImage(videoRef.current, 0, 0, 160, 120)
          frames.push(canvas.toDataURL("image/jpeg", 0.8))
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
    if (!stream) return null
    if (liveMode && capturing)
      return <motion.span animate={{ opacity: [1, 0.5, 1] }} transition={{ duration: 0.8, repeat: Infinity }}
        style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700,
          color: '#dc2626', background: 'rgba(220,38,38,0.08)', padding: '3px 8px', borderRadius: 20,
          border: '1px solid rgba(220,38,38,0.2)' }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#dc2626' }} />
        CAPTURING
      </motion.span>
    if (liveMode && processing)
      return <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700,
        color: '#d97706', background: 'rgba(217,119,6,0.08)', padding: '3px 8px', borderRadius: 20,
        border: '1px solid rgba(217,119,6,0.2)' }}>
        <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} />
        ANALYZING
      </span>
    if (liveMode && nextIn > 0)
      return <span style={{ fontSize: 11, fontWeight: 600, color: '#0d9488',
        background: 'rgba(13,148,136,0.08)', padding: '3px 8px', borderRadius: 20,
        border: '1px solid rgba(13,148,136,0.2)' }}>
        next in {nextIn}s
      </span>
    if (liveMode)
      return <motion.span animate={{ opacity: [1, 0.5, 1] }} transition={{ duration: 1, repeat: Infinity }}
        style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700,
          color: '#dc2626', background: 'rgba(220,38,38,0.08)', padding: '3px 8px', borderRadius: 20,
          border: '1px solid rgba(220,38,38,0.2)' }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#dc2626',
          boxShadow: '0 0 8px #dc2626' }} />
        LIVE
      </motion.span>
    return <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 600,
      color: '#059669', background: 'rgba(5,150,105,0.08)', padding: '3px 8px', borderRadius: 20,
      border: '1px solid rgba(5,150,105,0.2)' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#059669' }} />
      Camera On
    </span>
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Camera card */}
      <div style={{
        background: '#fff', borderRadius: 16, overflow: 'hidden',
        border: liveMode ? '1.5px solid rgba(220,38,38,0.25)' : '1px solid #e2e8f0',
        boxShadow: liveMode ? '0 4px 20px rgba(220,38,38,0.08)' : '0 1px 4px rgba(0,0,0,0.06)',
        transition: 'all 0.3s',
      }}>
        {/* Card header */}
        <div style={{ padding: '12px 14px', display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', borderBottom: '1px solid #f1f5f9', background: '#fafafa' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>Webcam</span>
          {statusBadge()}
          {!stream && <span style={{ fontSize: 11, color: '#94a3b8' }}>Camera off</span>}
        </div>

        {/* Video area */}
        <div style={{ position: 'relative', background: '#0f172a', aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          <canvas ref={canvasRef} width={640} height={480}
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />

          {!stream && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 12 }}>
              <div style={{ width: 56, height: 56, borderRadius: 16, background: 'rgba(255,255,255,0.06)',
                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CameraOff size={24} color="rgba(255,255,255,0.2)" />
              </div>
              <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.3)', fontWeight: 500 }}>Camera not started</p>
            </div>
          )}

          {/* Corner guides — idle state */}
          {stream && !liveMode && !capturing && !processing && [
            { top: 12, left: 12, borderTop: '2px solid rgba(13,148,136,0.5)', borderLeft: '2px solid rgba(13,148,136,0.5)' },
            { top: 12, right: 12, borderTop: '2px solid rgba(13,148,136,0.5)', borderRight: '2px solid rgba(13,148,136,0.5)' },
            { bottom: 12, left: 12, borderBottom: '2px solid rgba(13,148,136,0.5)', borderLeft: '2px solid rgba(13,148,136,0.5)' },
            { bottom: 12, right: 12, borderBottom: '2px solid rgba(13,148,136,0.5)', borderRight: '2px solid rgba(13,148,136,0.5)' },
          ].map((pos, i) => <div key={i} style={{ position: 'absolute', width: 18, height: 18, ...pos }} />)}

          {/* Countdown overlay */}
          <AnimatePresence>
            {capturing && !liveMode && countdown > 0 && (
              <motion.div key={countdown} initial={{ scale: 1.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.5, opacity: 0 }}
                style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ position: 'relative' }}>
                  <div style={{ width: 80, height: 80, borderRadius: '50%',
                    background: 'rgba(255,255,255,0.95)', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', boxShadow: '0 8px 32px rgba(0,0,0,0.3)' }}>
                    <span style={{ fontSize: 40, fontWeight: 900, color: '#dc2626' }}>{countdown}</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Analyzing overlay */}
          <AnimatePresence>
            {processing && !liveMode && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ position: 'absolute', inset: 0, background: 'rgba(15,23,42,0.75)',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                <div style={{ width: 52, height: 52, borderRadius: '50%',
                  background: 'rgba(13,148,136,0.15)', border: '1px solid rgba(13,148,136,0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Loader2 size={24} color="#0d9488" style={{ animation: 'spin 1s linear infinite' }} />
                </div>
                <p style={{ fontSize: 13, fontWeight: 600, color: 'white' }}>Analyzing sign...</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Live pulsing border */}
          {liveMode && capturing && (
            <div style={{ position: 'absolute', inset: 0, border: '2px solid rgba(220,38,38,0.5)',
              pointerEvents: 'none', animation: 'none' }} />
          )}

          {/* Progress bar */}
          {!liveMode && (capturing || processing) && (
            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 3, background: 'rgba(255,255,255,0.1)' }}>
              <motion.div animate={{ width: `${progress}%` }} transition={{ duration: 0.1 }}
                style={{ height: '100%', background: 'linear-gradient(to right, #0d9488, #059669)' }} />
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: stream ? '1fr 1fr 1fr' : '1fr', gap: 8 }}>
        {!stream ? (
          <motion.button onClick={startCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '13px 0', borderRadius: 12, border: 'none', cursor: 'pointer',
              background: 'linear-gradient(135deg, #0d9488, #059669)', color: 'white',
              fontSize: 13, fontWeight: 700, boxShadow: '0 4px 16px rgba(13,148,136,0.3)' }}>
            <Camera size={16} /> Start Camera
          </motion.button>
        ) : (
          <>
            <motion.button onClick={stopCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                padding: '11px 0', borderRadius: 12, cursor: 'pointer', fontSize: 13, fontWeight: 600,
                background: '#f8fafc', border: '1px solid #e2e8f0', color: '#64748b',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <Square size={13} /> Stop
            </motion.button>

            <motion.button onClick={toggleLive} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                padding: '11px 0', borderRadius: 12, cursor: 'pointer',
                background: liveMode ? 'linear-gradient(135deg, #dc2626, #ef4444)' : '#fff',
                border: liveMode ? 'none' : '1px solid rgba(220,38,38,0.3)',
                color: liveMode ? 'white' : '#dc2626',
                fontSize: 13, fontWeight: 700,
                boxShadow: liveMode ? '0 4px 14px rgba(220,38,38,0.3)' : '0 1px 3px rgba(0,0,0,0.05)' }}>
              <Radio size={13} />
              {liveMode ? "Stop Live" : "Go Live"}
            </motion.button>

            <motion.button onClick={startManualCapture}
              disabled={capturing || processing || liveMode}
              whileHover={!capturing && !processing && !liveMode ? { scale: 1.02 } : {}}
              whileTap={!capturing && !processing && !liveMode ? { scale: 0.97 } : {}}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                padding: '11px 0', borderRadius: 12,
                cursor: capturing || processing || liveMode ? 'not-allowed' : 'pointer',
                background: capturing || processing || liveMode
                  ? '#f8fafc' : 'linear-gradient(135deg, #0d9488, #059669)',
                border: capturing || processing || liveMode ? '1px solid #e2e8f0' : 'none',
                color: capturing || processing || liveMode ? '#94a3b8' : 'white',
                fontSize: 13, fontWeight: 700, opacity: liveMode ? 0.5 : 1,
                boxShadow: capturing || processing || liveMode ? 'none' : '0 4px 14px rgba(13,148,136,0.3)' }}>
              {processing && !liveMode
                ? <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />
                : <Zap size={13} />}
              {capturing && !liveMode ? `${countdown}s...` : "Capture"}
            </motion.button>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: 'rgba(220,38,38,0.06)', border: '1px solid rgba(220,38,38,0.2)',
          borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#dc2626', fontWeight: 500 }}>
          ⚠ {error}
        </div>
      )}

      {/* Hint */}
      <div style={{ background: '#fff', border: '1px solid #f1f5f9', borderRadius: 10,
        padding: '10px 14px', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
        <p style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.7 }}>
          <span style={{ color: '#dc2626', fontWeight: 700 }}>Go Live</span> — auto-predicts every ~5s with overlay on feed
          {"  ·  "}
          <span style={{ color: '#0d9488', fontWeight: 700 }}>Capture</span> — manual 3s recording
        </p>
      </div>
    </div>
  )
}
