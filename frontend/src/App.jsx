import { useState, useCallback } from 'react'
import Header from './components/Header'
import SignSelector from './components/SignSelector'
import WebcamCapture from './components/WebcamCapture'
import PredictionPanel from './components/PredictionPanel'
import { useSystemStatus } from './hooks/useSystemStatus'

export default function App() {
  const [selectedSign, setSelectedSign]   = useState(null)
  const [prediction, setPrediction]       = useState(null)
  const [translation, setTranslation]     = useState('')
  const [translating, setTranslating]     = useState(false)
  const [sentence, setSentence]           = useState([])   // [{word, conf, translation, id}]
  const [history, setHistory]             = useState([])   // past spoken sentences
  const [isSpeaking, setIsSpeaking]       = useState(false)
  const [speakingTarget, setSpeakingTarget] = useState(null) // 'word' | 'sentence'
  const status = useSystemStatus()

  // ── Called by WebcamCapture after every successful /api/predict ────────────
  const onPrediction = useCallback((data) => {
    setPrediction(data)
    setTranslation('')
    setTranslating(true)
    fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: data.top_label })
    })
      .then(r => r.json())
      .then(d => {
        setTranslating(false)
        if (!d.error) setTranslation(d.translation)
      })
      .catch(() => setTranslating(false))
  }, [])

  // ── Add current prediction to the sentence queue ───────────────────────────
  const addToSentence = useCallback((word, conf, trans) => {
    setSentence(prev => [
      ...prev,
      { id: Date.now(), word, conf, translation: trans }
    ])
  }, [])

  // ── Remove one word chip by id ─────────────────────────────────────────────
  const removeFromSentence = useCallback((id) => {
    setSentence(prev => prev.filter(w => w.id !== id))
  }, [])

  // ── Clear whole sentence ───────────────────────────────────────────────────
  const clearSentence = useCallback(() => setSentence([]), [])

  // ── TTS helper — plays a wav blob, returns a Promise that resolves on end ──
  const playTTS = async (text) => {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    const data = await res.json()
    if (!data.audio_b64) throw new Error('No audio returned')
    const bytes = atob(data.audio_b64)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const blob = new Blob([buf], { type: 'audio/wav' })
    const url  = URL.createObjectURL(blob)
    return new Promise((resolve, reject) => {
      const audio = new Audio(url)
      audio.play().catch(reject)
      audio.onended = () => { URL.revokeObjectURL(url); resolve() }
      audio.onerror = () => { URL.revokeObjectURL(url); reject() }
    })
  }

  // ── Speak the single current word translation ──────────────────────────────
  const onSpeakWord = async () => {
    if (!translation || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('word')
    try { await playTTS(translation) } catch { /* silent */ }
    setIsSpeaking(false); setSpeakingTarget(null)
  }

  // ── Speak the full sentence queue ──────────────────────────────────────────
  const onSpeakSentence = async () => {
    if (!sentence.length || isSpeaking) return
    const fullText = sentence.map(w => w.translation).join(' ')
    const englishText = sentence.map(w => w.word).join(' ')
    setIsSpeaking(true); setSpeakingTarget('sentence')
    try {
      await playTTS(fullText)
      // Log to history as a complete sentence
      setHistory(prev => [
        {
          sentence: englishText,
          kannada: fullText,
          wordCount: sentence.length,
          time: new Date().toLocaleTimeString()
        },
        ...prev.slice(0, 9)
      ])
    } catch { /* silent */ }
    setIsSpeaking(false); setSpeakingTarget(null)
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
          translating={translating}
          sentence={sentence}
          history={history}
          isSpeaking={isSpeaking}
          speakingTarget={speakingTarget}
          onSpeakWord={onSpeakWord}
          onSpeakSentence={onSpeakSentence}
          onAddToSentence={addToSentence}
          onRemoveFromSentence={removeFromSentence}
          onClearSentence={clearSentence}
        />
      </main>
    </div>
  )
}
