import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, CircleStop, Loader2, CameraOff } from 'lucide-react'

export default function WebcamCapture({ selectedSign, onPrediction }) {
  const videoRef = useRef(null)
  const [stream, setStream] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [countdown, setCountdown] = useState(0)
  const [error, setError] = useState('')

  const startCamera = async () => {
    setError('')
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      setStream(s)
      if (videoRef.current) videoRef.current.srcObject = s
    } catch (e) {
      setError('Camera access denied: ' + e.message)
    }
  }

  const stopCamera = () => {
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
  }

  const startCapture = () => {
    if (!stream || capturing || processing) return
    setCapturing(true)
    setCountdown(3)

    const frames = []
    const canvas = document.createElement('canvas')
    canvas.width = 320; canvas.height = 240
    const ctx = canvas.getContext('2d')

    const DURATION = 3000
    const INTERVAL = 100
    let elapsed = 0

    const cdInterval = setInterval(() => {
      elapsed += 100
      setCountdown(Math.ceil((DURATION - elapsed) / 1000))
      ctx.drawImage(videoRef.current, 0, 0, 320, 240)
      frames.push(canvas.toDataURL('image/jpeg', 0.7))
      if (elapsed >= DURATION) {
        clearInterval(cdInterval)
        setCapturing(false)
        setCountdown(0)
        setProcessing(true)
        predict(frames)
      }
    }, INTERVAL)
  }

  const predict = async (frames) => {
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames })
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      onPrediction(data)
    } catch (e) {
      setError('Prediction failed: ' + e.message)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-2xl border border-white/5 bg-[#0d0d14] overflow-hidden">
        {/* Webcam header */}
        <div className="px-4 pt-3 pb-2 flex items-center justify-between">
          <h2 className="text-[13px] font-semibold text-white">Your Webcam</h2>
          {stream ? (
            <div className="flex items-center gap-1.5 text-[11px] text-green-400">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Live
            </div>
          ) : (
            <span className="text-[11px] text-zinc-600">Camera off</span>
          )}
        </div>

        {/* Video area */}
        <div className="relative bg-black aspect-video">
          <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />

          {!stream && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <CameraOff size={32} className="text-zinc-700" />
              <p className="text-zinc-600 text-[13px]">Camera not started</p>
            </div>
          )}

          {/* Countdown overlay */}
          <AnimatePresence>
            {capturing && countdown > 0 && (
              <motion.div
                key={countdown}
                initial={{ scale: 1.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.5, opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <div className="w-24 h-24 rounded-full bg-black/70 border-4 border-red-500 flex items-center justify-center">
                  <span className="text-5xl font-black text-white">{countdown}</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing overlay */}
          <AnimatePresence>
            {processing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center gap-3"
              >
                <Loader2 size={32} className="text-violet-400 animate-spin" />
                <p className="text-[13px] text-violet-300">Analyzing sign...</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* REC badge */}
          {capturing && (
            <div className="absolute top-3 right-3 bg-red-500 text-white text-[11px] font-bold px-2.5 py-1 rounded-full flex items-center gap-1.5 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-white" />
              REC
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-2">
        {!stream ? (
          <motion.button
            onClick={startCamera}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            className="col-span-2 flex items-center justify-center gap-2 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-[13px] font-semibold transition-colors"
          >
            <Camera size={15} />
            Start Camera
          </motion.button>
        ) : (
          <>
            <motion.button
              onClick={stopCamera}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              className="flex items-center justify-center gap-2 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[13px] font-medium transition-colors border border-white/5"
            >
              <CircleStop size={14} />
              Stop
            </motion.button>
            <motion.button
              onClick={startCapture}
              disabled={capturing || processing}
              whileHover={!capturing && !processing ? { scale: 1.02 } : {}}
              whileTap={!capturing && !processing ? { scale: 0.97 } : {}}
              className="flex items-center justify-center gap-2 py-3 rounded-xl bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-[13px] font-semibold transition-colors"
            >
              {processing ? <Loader2 size={14} className="animate-spin" /> : '▶'}
              {capturing ? `Capturing (${countdown}s)` : processing ? 'Processing...' : 'Capture (3s)'}
            </motion.button>
          </>
        )}
      </div>

      {error && (
        <div className="rounded-xl bg-red-950/30 border border-red-500/20 px-4 py-2.5 text-[12px] text-red-400">
          {error}
        </div>
      )}

      <div className="rounded-xl bg-white/2 border border-white/5 px-4 py-3">
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          <span className="text-zinc-300 font-medium">How to use:</span> Start camera → select a sign → watch the reference video → perform the same sign in front of camera → click Capture
        </p>
      </div>
    </div>
  )
}
