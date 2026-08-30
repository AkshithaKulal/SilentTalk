import { useState, useCallback, useRef, useEffect } from 'react'
import Header from './components/Header'
import SignSelector from './components/SignSelector'
import WebcamCapture from './components/WebcamCapture'
import PredictionPanel from './components/PredictionPanel'
import { useSystemStatus } from './hooks/useSystemStatus'

export default function App() {
  const [selectedSign, setSelectedSign]     = useState(null)
  const [prediction, setPrediction]         = useState(null)
  const [translation, setTranslation]       = useState('')
  const [translating, setTranslating]       = useState(false)
  const [sentence, setSentence]             = useState([])   // [{word, conf, translation, id}]
  const [history, setHistory]               = useState([])
  const [isSpeaking, setIsSpeaking]         = useState(false)
  const [speakingTarget, setSpeakingTarget] = useState(null)
  const [selectedVoice, setSelectedVoice]   = useState('female_clear')
  const [voices, setVoices]                 = useState([])
  const status = useSystemStatus()

  // ── Client-side translation cache (Fix 3) ─────────────────────────────────
  // Word → Kannada translation. Avoids even the HTTP call for repeated words.
  const translationCache = useRef({})

  // Load voices on mount
  useEffect(() => {
    fetch('/api/voices').then(r => r.json()).then(d => {
      if (d.voices) setVoices(d.voices)
    }).catch(() => {})
  }, [])

  // ── Translate one word — cache-first (Fix 3) ───────────────────────────────
  const translateWord = useCallback(async (word) => {
    const key = word.toLowerCase().trim()
    if (translationCache.current[key]) {
      return translationCache.current[key]
    }
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: word })
    })
    const d = await res.json()
    if (d.translation) {
      translationCache.current[key] = d.translation
      return d.translation
    }
    return word   // fallback to English if translation fails
  }, [])

  // ── FIX 1: onPrediction no longer auto-translates ─────────────────────────
  // Translation ONLY happens when the user explicitly adds the word to sentence
  // or clicks Speak. This stops the GPU from being hammered on every live frame.
  const onPrediction = useCallback((data) => {
    setPrediction(data)
    setTranslation('')
    setTranslating(false)
  }, [])

  // ── Add to sentence: translate on demand (Fix 1 + Fix 3) ─────────────────
  const addToSentence = useCallback(async (word, conf) => {
    setTranslating(true)
    try {
      const trans = await translateWord(word)
      setTranslation(trans)   // show in prediction panel
      setSentence(prev => [
        ...prev,
        { id: Date.now(), word, conf, translation: trans }
      ])
    } finally {
      setTranslating(false)
    }
  }, [translateWord])

  const removeFromSentence = useCallback((id) => {
    setSentence(prev => prev.filter(w => w.id !== id))
  }, [])

  const clearSentence = useCallback(() => setSentence([]), [])

  // ── Speak single word: fast mms (<0.5s) ─────────────────────────────────
  const onSpeakWord = useCallback(async () => {
    if (!prediction || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('word')
    try {
      const trans = await translateWord(prediction.top_label)
      setTranslation(trans)
      await playTTS(trans, true)   // fast=true → mms-tts-kan, instant
    } catch { /* silent */ }
    setIsSpeaking(false); setSpeakingTarget(null)
  }, [prediction, isSpeaking, selectedVoice])

  // ── Speak sentence: batch translate uncached words in ONE GPU call (Fix 2) ─
  const onSpeakSentence = useCallback(async () => {
    if (!sentence.length || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('sentence')

    try {
      // Find words that still need translation (not in client cache)
      const wordsNeedingTranslation = sentence
        .filter(w => !translationCache.current[w.word.toLowerCase().trim()])
        .map(w => w.word)

      if (wordsNeedingTranslation.length > 0) {
        // FIX 2: single batch call for all uncached words
        const res = await fetch('/api/translate_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ words: wordsNeedingTranslation })
        })
        const d = await res.json()
        if (d.pairs) {
          d.pairs.forEach(({ word, translation }) => {
            translationCache.current[word.toLowerCase().trim()] = translation
          })
        }
      }

      // Now all words are cached — build full sentence
      const updatedSentence = sentence.map(w => ({
        ...w,
        translation: translationCache.current[w.word.toLowerCase().trim()] || w.translation
      }))

      const fullKannada  = updatedSentence.map(w => w.translation).join(' ')
      const fullEnglish  = updatedSentence.map(w => w.word).join(' ')

      // Update sentence state with fresh translations
      setSentence(updatedSentence)

      await playTTS(fullKannada, false)   // fast=false → parler quality for sentences

      setHistory(prev => [
        {
          sentence: fullEnglish,
          kannada:  fullKannada,
          wordCount: updatedSentence.length,
          time: new Date().toLocaleTimeString()
        },
        ...prev.slice(0, 9)
      ])
      setSentence([])   // auto-clear after speaking
    } catch { /* silent */ }

    setIsSpeaking(false); setSpeakingTarget(null)
  }, [sentence, isSpeaking, selectedVoice])

  // ── Replay from history ───────────────────────────────────────────────────
  const onReplayHistory = useCallback(async (kannada) => {
    if (!kannada || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('replay')
    try { await playTTS(kannada, false) } catch { /* silent */ }  // quality replay
    setIsSpeaking(false); setSpeakingTarget(null)
  }, [isSpeaking, selectedVoice])

  // ── TTS helper ────────────────────────────────────────────────────────────
  const playTTS = async (text, fast = false) => {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: selectedVoice, fast })
    })
    const data = await res.json()
    if (!data.audio_b64) throw new Error('No audio')
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

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#e8edf3' }}>
      <Header status={status} />
      <main style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '300px 1fr 340px',
        gap: 16,
        padding: 16,
        maxWidth: 1600,
        margin: '0 auto',
        width: '100%',
        alignItems: 'start',
        minHeight: 0,
      }}>
        <div style={{ minHeight: 0, position: 'sticky', top: 76, maxHeight: 'calc(100vh - 92px)', overflowY: 'auto' }}>
          <SignSelector onSelect={setSelectedSign} selected={selectedSign} />
        </div>

        <div style={{ minHeight: 0 }}>
          <WebcamCapture selectedSign={selectedSign} onPrediction={onPrediction} />
        </div>

        <div style={{ minHeight: 0, position: 'sticky', top: 76, maxHeight: 'calc(100vh - 92px)', overflowY: 'auto' }}>
          <PredictionPanel
            prediction={prediction}
            translation={translation}
            translating={translating}
            sentence={sentence}
            history={history}
            isSpeaking={isSpeaking}
            speakingTarget={speakingTarget}
            voices={voices}
            selectedVoice={selectedVoice}
            onVoiceChange={setSelectedVoice}
            onSpeakWord={onSpeakWord}
            onSpeakSentence={onSpeakSentence}
            onReplayHistory={onReplayHistory}
            onAddToSentence={addToSentence}
            onRemoveFromSentence={removeFromSentence}
            onClearSentence={clearSentence}
          />
        </div>
      </main>
    </div>
  )
}
