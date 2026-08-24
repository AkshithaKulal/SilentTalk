import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, Square, Loader2, CameraOff, Zap } from 'lucide-react'

export default function WebcamCapture({ selectedSign, onPrediction }) {
  const videoRef = useRef(null)
  const [stream, setStream] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [countdown, setCountdown] = useState(0)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  const startCamera = async () => {
    setError('')
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      setStream(s)
      if (videoRef.current) videoRef.current.srcObject = s
    } catch (e) { setError('Camera access denied: ' + e.message) }
  }

  const stopCamera = () => {
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
  }

  const startCapture = () => {
    if (!stream || capturing || processing) return
    setCapturing(true)
    setCountdown(3)
    setProgress(0)

    const frames = []
    const canvas = document.createElement('canvas')
    canvas.width = 320; canvas.height = 240
    const ctx = canvas.getContext('2d')
    const DURATION = 3000, INTERVAL = 100
    let elapsed = 0

    const interval = setInterval(() => {
      elapsed += INTERVAL
      setProgress(elapsed / DURATION * 100)
      setCountdown(Math.ceil((DURATION - elapsed) / 1000))
      ctx.drawImage(videoRef.current, 0, 0, 320, 240)
      frames.push(canvas.toDataURL('image/jpeg', 0.7))
      if (elapsed >= DURATION) {
        clearInterval(interval)
        setCapturing(false); setCountdown(0); setProgress(100)
        setProcessing(true)
        predict(frames)
      }
    }, INTERVAL)
  }

  const predict = async (frames) => {
    try {
      const res = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames })
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      onPrediction(data)
    } catch (e) { setError('Prediction failed: ' + e.message) }
    finally { setProcessing(false); setProgress(0) }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Camera card */}
      <div style={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 16, overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>Your Webcam</span>
          {stream
            ? <span style={{ fontSize: 11, color: '#34d399', display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#34d399',
                  boxShadow: '0 0 8px #34d399', animation: 'pulse 1.5s infinite' }} />
                Live
              </span>
            : <span style={{ fontSize: 11, color: '#334155' }}>Camera off</span>
          }
        </div>

        {/* Video area */}
        <div style={{ position: 'relative', background: '#060912', aspectRatio: '4/3' }}>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />

          {!stream && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 12 }}>
              <CameraOff size={36} color="#1e293b" />
              <p style={{ fontSize: 13, color: '#334155' }}>Camera not started</p>
            </div>
          )}

          {/* Corner guides when active */}
          {stream && !capturing && !processing && (
            <>
              {[['0','0'],['0','auto'],['auto','0'],['auto','auto']].map(([t,b], i) => (
                <div key={i} style={{
                  position: 'absolute',
                  top: t === '0' ? 12 : 'auto', bottom: b === 'auto' && t === 'auto' ? 12 : t === 'auto' ? 12 : 'auto',
                  left: i < 2 ? 12 : 'auto', right: i >= 2 ? 12 : 'auto',
                  width: 16, height: 16,
                  borderTop: i < 2 ? '2px solid rgba(99,102,241,0.5)' : 'none',
                  borderBottom: i >= 2 ? '2px solid rgba(99,102,241,0.5)' : 'none',
                  borderLeft: i % 2 === 0 ? '2px solid rgba(99,102,241,0.5)' : 'none',
                  borderRight: i % 2 === 1 ? '2px solid rgba(99,102,241,0.5)' : 'none',
                }} />
              ))}
            </>
          )}

          {/* Countdown */}
          <AnimatePresence>
            {capturing && countdown > 0 && (
              <motion.div key={countdown}
                initial={{ scale: 1.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.4, opacity: 0 }}
                style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 80, height: 80, borderRadius: '50%',
                  background: 'rgba(0,0,0,0.75)', border: '3px solid #ef4444',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 0 30px rgba(239,68,68,0.4)' }}>
                  <span style={{ fontSize: 42, fontWeight: 900, color: 'white' }}>{countdown}</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing overlay */}
          <AnimatePresence>
            {processing && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ position: 'absolute', inset: 0, background: 'rgba(6,9,18,0.88)',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                <Loader2 size={28} color="#818cf8" style={{ animation: 'spin 1s linear infinite' }} />
                <p style={{ fontSize: 13, color: '#818cf8', fontWeight: 600 }}>Analyzing sign...</p>
                <p style={{ fontSize: 11, color: '#475569' }}>Running MediaPipe + classifier</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* REC badge */}
          {capturing && (
            <div style={{ position: 'absolute', top: 10, right: 10,
              background: '#ef4444', color: 'white', fontSize: 11, fontWeight: 700,
              padding: '3px 10px', borderRadius: 20, display: 'flex', alignItems: 'center', gap: 5,
              boxShadow: '0 0 12px rgba(239,68,68,0.5)', animation: 'pulse 1s infinite' }}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'white' }} />
              REC
            </div>
          )}

          {/* Progress bar at bottom */}
          {(capturing || processing) && (
            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 3,
              background: 'rgba(255,255,255,0.05)' }}>
              <motion.div animate={{ width: `${progress}%` }} transition={{ duration: 0.1 }}
                style={{ height: '100%', background: 'linear-gradient(to right, #6366f1, #a78bfa)' }} />
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {!stream ? (
          <motion.button onClick={startCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            style={{
              gridColumn: '1 / -1', display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: 8, padding: '11px 0', borderRadius: 12, border: 'none', cursor: 'pointer',
              background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
              color: 'white', fontSize: 13, fontWeight: 700,
              boxShadow: '0 4px 20px rgba(99,102,241,0.3)',
            }}>
            <Camera size={15} /> Start Camera
          </motion.button>
        ) : (
          <>
            <motion.button onClick={stopCamera} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
                padding: '11px 0', borderRadius: 12, cursor: 'pointer', fontSize: 13, fontWeight: 600,
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', color: '#64748b'
              }}>
              <Square size={13} /> Stop
            </motion.button>
            <motion.button
              onClick={startCapture}
              disabled={capturing || processing}
              whileHover={!capturing && !processing ? { scale: 1.02 } : {}}
              whileTap={!capturing && !processing ? { scale: 0.97 } : {}}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
                padding: '11px 0', borderRadius: 12, border: 'none', cursor: capturing || processing ? 'not-allowed' : 'pointer',
                background: capturing || processing ? 'rgba(34,197,94,0.2)' : 'linear-gradient(135deg, #16a34a, #22c55e)',
                color: 'white', fontSize: 13, fontWeight: 700, opacity: capturing || processing ? 0.7 : 1,
                boxShadow: capturing || processing ? 'none' : '0 4px 16px rgba(34,197,94,0.25)',
              }}>
              {processing ? <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} /> : <Zap size={13} />}
              {capturing ? `${countdown}s...` : processing ? 'Analyzing...' : 'Capture (3s)'}
            </motion.button>
          </>
        )}
      </div>

      {error && (
        <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
          borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#fca5a5' }}>
          {error}
        </div>
      )}

      <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
        borderRadius: 10, padding: '10px 14px' }}>
        <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.6 }}>
          <span style={{ color: '#94a3b8', fontWeight: 600 }}>How to use: </span>
          Start camera → select a sign from the left → watch the reference video → perform the same sign → click Capture
        </p>
      </div>
    </div>
  )
}
