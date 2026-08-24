import { useState, useEffect } from 'react'
import Header from './components/Header'
import SignSelector from './components/SignSelector'
import WebcamCapture from './components/WebcamCapture'
import PredictionPanel from './components/PredictionPanel'
import { useSystemStatus } from './hooks/useSystemStatus'

export default function App() {
  const [selectedSign, setSelectedSign] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [translation, setTranslation] = useState('')
  const [history, setHistory] = useState([])
  const [isSpeaking, setIsSpeaking] = useState(false)
  const status = useSystemStatus()

  const onPrediction = (data) => {
    setPrediction(data)
    setTranslation('')
    // Auto-translate top prediction
    fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: data.top_label })
    })
      .then(r => r.json())
      .then(d => {
        if (!d.error) {
          setTranslation(d.translation)
          setHistory(prev => [
            { label: data.top_label, conf: data.top_conf, translation: d.translation, time: new Date().toLocaleTimeString() },
            ...prev.slice(0, 19)
          ])
        }
      })
  }

  const onSpeak = async () => {
    if (!translation || isSpeaking) return
    setIsSpeaking(true)
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: translation })
      })
      const data = await res.json()
      if (data.audio_b64) {
        const bytes = atob(data.audio_b64)
        const buf = new Uint8Array(bytes.length)
        for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
        const blob = new Blob([buf], { type: 'audio/wav' })
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        audio.play()
        audio.onended = () => { URL.revokeObjectURL(url); setIsSpeaking(false) }
      }
    } catch { setIsSpeaking(false) }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#060912' }}>
      <Header status={status} />
      <main style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 16,
        padding: 16,
        maxWidth: 1600,
        margin: '0 auto',
        width: '100%',
        alignItems: 'start',
      }}>
        <SignSelector onSelect={setSelectedSign} selected={selectedSign} />
        <WebcamCapture selectedSign={selectedSign} onPrediction={onPrediction} />
        <PredictionPanel
          prediction={prediction}
          translation={translation}
          history={history}
          onSpeak={onSpeak}
          isSpeaking={isSpeaking}
        />
      </main>
    </div>
  )
}
