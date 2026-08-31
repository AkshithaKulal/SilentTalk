import { useState, useCallback, useRef, useEffect } from 'react'
import Header from './components/Header'
import SignSelector from './components/SignSelector'
import WebcamCapture from './components/WebcamCapture'
import MessageBar from './components/MessageBar'
import LiveRail from './components/LiveRail'
import { useSystemStatus } from './hooks/useSystemStatus'
import { DEFAULT_VOICE, DEFAULT_ENGINE, VOICE_SAMPLE_KN, voiceById } from './voices'

export default function App() {
  const [libraryOpen, setLibraryOpen]       = useState(false)
  const [selectedSign, setSelectedSign]     = useState(null)
  const [prediction, setPrediction]         = useState(null)
  const [translation, setTranslation]       = useState('')
  const [translating, setTranslating]       = useState(false)
  const [sentence, setSentence]             = useState([])   // [{word, conf, translation, id}]
  const [history, setHistory]               = useState([])
  const [isSpeaking, setIsSpeaking]         = useState(false)
  const [speakingTarget, setSpeakingTarget] = useState(null)
  const [selectedVoice, setSelectedVoice]   = useState(() => {
    try { return localStorage.getItem('st-voice') || DEFAULT_VOICE } catch { return DEFAULT_VOICE }
  })
  const [selectedEngine, setSelectedEngine]   = useState(() => {
    try { return localStorage.getItem('st-engine') || DEFAULT_ENGINE } catch { return DEFAULT_ENGINE }
  })
  const [parlerReady, setParlerReady]       = useState(false)
  const [sarvamReady, setSarvamReady]         = useState(false)
  const status = useSystemStatus()

  // Auto-commit: same gloss must stay confident across live windows, then
  // it is appended — no "Add to Sentence" click. Cooldown stops duplicates.
  const pendingRef = useRef({ label: null, count: 0 })
  const lastCommitRef = useRef({ label: null, t: 0 })
  const AUTO_CONF = 58
  const SAME_WORD_COOLDOWN_MS = 2800

  // ── Client-side translation cache (Fix 3) ─────────────────────────────────
  // Word → Kannada translation. Avoids even the HTTP call for repeated words.
  const translationCache = useRef({})

  // Load voices on mount
  useEffect(() => {
    fetch('/api/voices').then(r => r.json()).then(d => {
      if (typeof d.parler_ready === 'boolean') setParlerReady(d.parler_ready)
      if (typeof d.sarvam_ready === 'boolean') setSarvamReady(d.sarvam_ready)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    try { localStorage.setItem('st-engine', selectedEngine) } catch { /* ignore */ }
  }, [selectedEngine])

  useEffect(() => {
    try { localStorage.setItem('st-voice', selectedVoice) } catch { /* ignore */ }
  }, [selectedVoice])

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

  // Append immediately (English chip). Kannada fills in without blocking Speak.
  const addToSentence = useCallback(async (word, conf) => {
    const id = Date.now() + Math.random()
    const key = word.toLowerCase().trim()
    const cached = translationCache.current[key] || ''
    setSentence(prev => [...prev, { id, word, conf, translation: cached }])
    if (cached) {
      setTranslation(cached)
      return
    }
    setTranslating(true)
    try {
      const trans = await translateWord(word)
      setTranslation(trans)
      setSentence(prev => prev.map(w => w.id === id ? { ...w, translation: trans } : w))
    } finally {
      setTranslating(false)
    }
  }, [translateWord])

  // Live predictions auto-commit. Translation still only on add/Speak, not every frame.
  const onPrediction = useCallback((data) => {
    setPrediction(data)
    if (data?.idle) {
      pendingRef.current = { label: null, count: 0 }
      return
    }
    const conf = data?.top_conf ?? 0
    const margin = data?.margin ?? 100
    const label = (data?.top_label || '').trim()
    // Soft gate: still need some confidence; margin only blocks near-ties
    if (!label || conf < AUTO_CONF || (conf < 70 && margin < 5)) {
      pendingRef.current = { label: null, count: 0 }
      return
    }
    if (pendingRef.current.label === label) {
      pendingRef.current.count += 1
    } else {
      pendingRef.current = { label, count: 1 }
    }
    const needStreak = conf >= 75 ? 1 : 2
    const now = Date.now()
    const sameAsLast = lastCommitRef.current.label === label
    const tooSoon = now - lastCommitRef.current.t < SAME_WORD_COOLDOWN_MS
    if (pendingRef.current.count >= needStreak && !(sameAsLast && tooSoon)) {
      lastCommitRef.current = { label, t: now }
      pendingRef.current = { label, count: 0 }
      addToSentence(label, conf)
    }
  }, [addToSentence])

  const removeFromSentence = useCallback((id) => {
    setSentence(prev => prev.filter(w => w.id !== id))
  }, [])

  const undoLast = useCallback(() => {
    setSentence(prev => prev.slice(0, -1))
  }, [])

  const clearSentence = useCallback(() => setSentence([]), [])

  // ── Speak single word: fast mms (<0.5s) ─────────────────────────────────
  const onSpeakWord = useCallback(async () => {
    if (!prediction || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('word')
    try {
      const trans = await translateWord(prediction.top_label)
      setTranslation(trans)
      await playTTS(trans)
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

      // Always use MMS path for reliable Speak (Parler cache may be corrupt)
      await playTTS(fullKannada)

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
  }, [sentence, isSpeaking, selectedVoice, parlerReady])

  // ── Replay from history ───────────────────────────────────────────────────
  const onReplayHistory = useCallback(async (kannada) => {
    if (!kannada || isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('replay')
    try { await playTTS(kannada) } catch { /* silent */ }
    setIsSpeaking(false); setSpeakingTarget(null)
  }, [isSpeaking, selectedVoice])

  const onPreviewVoice = useCallback(async () => {
    if (isSpeaking) return
    setIsSpeaking(true); setSpeakingTarget('preview')
    try { await playTTS(VOICE_SAMPLE_KN) } catch { /* silent */ }
    setIsSpeaking(false); setSpeakingTarget(null)
  }, [isSpeaking, selectedVoice, parlerReady])

  const playTTS = async (text) => {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: selectedVoice, engine: selectedEngine })
    })
    const data = await res.json()
    if (!res.ok || !data.audio_b64) {
      throw new Error(data.detail || data.error || 'TTS failed')
    }
    const mime = data.format === 'mp3' ? 'audio/mpeg' : 'audio/wav'
    const bytes = atob(data.audio_b64)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const blob = new Blob([buf], { type: mime })
    const url  = URL.createObjectURL(blob)
    return new Promise((resolve, reject) => {
      const audio = new Audio(url)
      audio.play().catch(reject)
      audio.onended = () => { URL.revokeObjectURL(url); resolve() }
      audio.onerror = () => { URL.revokeObjectURL(url); reject() }
    })
  }

  const onSpeakTap = useCallback(async () => {
    if (isSpeaking) return
    if (sentence.length > 0) {
      await onSpeakSentence()
    } else if (prediction?.top_label && !prediction?.idle) {
      await onSpeakWord()
    }
  }, [sentence.length, prediction, isSpeaking, onSpeakSentence, onSpeakWord])

  const canSpeakNow =
    !isSpeaking &&
    (sentence.length > 0 || (prediction?.top_label && !prediction?.idle))

  const speakHint =
    sentence.length > 0
      ? "Tap Speak for Kannada audio"
      : prediction?.top_label && !prediction?.idle
        ? `Tap Speak for “${prediction.top_label}”`
        : "Hold a sign — Speak lights up when a word is detected"

  return (
    <div className="app-shell">
      <Header
        status={status}
        libraryOpen={libraryOpen}
        onToggleLibrary={() => setLibraryOpen((v) => !v)}
        selectedVoice={selectedVoice}
        selectedEngine={selectedEngine}
        onVoiceChange={setSelectedVoice}
        onEngineChange={setSelectedEngine}
        onPreviewVoice={onPreviewVoice}
        isPreviewing={speakingTarget === 'preview' && isSpeaking}
        parlerReady={parlerReady}
        sarvamReady={sarvamReady}
        voiceBusy={isSpeaking}
      />
      <main className="workspace">
        <WebcamCapture
          selectedSign={selectedSign}
          onPrediction={onPrediction}
          onSpeakNow={onSpeakWord}
          canSpeakNow={canSpeakNow && !isSpeaking && !!prediction?.top_label && !prediction?.idle && sentence.length === 0}
          isSpeaking={isSpeaking}
        />
        <LiveRail
          history={history}
          isSpeaking={isSpeaking}
          speakingTarget={speakingTarget}
          onReplayHistory={onReplayHistory}
        />
      </main>
      <MessageBar
        sentence={sentence}
        isSpeaking={(speakingTarget === 'sentence' || speakingTarget === 'preview' || speakingTarget === 'word') && isSpeaking}
        speakingName={
          speakingTarget === 'sentence'
            ? voiceById(selectedVoice).name
            : speakingTarget === 'preview'
              ? `Trying ${voiceById(selectedVoice).name}`
              : speakingTarget === 'word'
                ? prediction?.top_label || 'Speaking'
                : ''
        }
        onSpeak={onSpeakTap}
        canSpeak={canSpeakNow}
        speakHint={speakHint}
        onClear={clearSentence}
        onRemove={removeFromSentence}
        onUndo={undoLast}
      />
      {libraryOpen && (
        <SignSelector
          onSelect={setSelectedSign}
          selected={selectedSign}
          onClose={() => setLibraryOpen(false)}
        />
      )}
    </div>
  )
}
