import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { Check, Loader2, Volume2 } from "lucide-react"
import { VOICES, TTS_ENGINES, voiceById, engineById } from "../voices"

export default function VoicePicker({
  selectedVoice,
  selectedEngine,
  onVoiceChange,
  onEngineChange,
  onPreview,
  isPreviewing,
  sarvamReady,
  parlerReady,
  disabled,
}) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef(null)
  const current = voiceById(selectedVoice)
  const engine = engineById(selectedEngine)

  useEffect(() => {
    const onDoc = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("mousedown", onDoc)
    window.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onDoc)
      window.removeEventListener("keydown", onKey)
    }
  }, [])

  const engineNote =
    selectedEngine === "sarvam"
      ? sarvamReady
        ? "Sarvam Bulbul — best Kannada pronunciation (cloud)."
        : "Add SARVAM_API_KEY to .env on the server, then restart."
      : selectedEngine === "parler"
        ? parlerReady
          ? "Indic Parler runs on your GPU (~2 GB)."
          : "Parler not loaded — will fall back to MMS."
        : selectedEngine === "mms"
          ? "MMS — one fixed voice, fastest, lowest quality."
          : sarvamReady
            ? "Auto uses Sarvam first, then Parler, then MMS."
            : parlerReady
              ? "Auto uses Parler, then MMS."
              : "Auto uses MMS until Sarvam or Parler is available."

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="listbox"
        className="header-btn"
        title="Voice and TTS engine"
      >
        <span className="voice-dot" style={{ background: current.tone }} />
        {current.name}
        <span style={{ color: "var(--faint)", fontWeight: 600, fontSize: 11 }}>
          · {engine.name}
        </span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="listbox"
            aria-label="Speaking voice"
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 420, damping: 28 }}
            className="voice-pop"
          >
            <div className="voice-pop-title">Speech engine</div>
            <div className="engine-list">
              {TTS_ENGINES.map((e) => {
                const on = e.id === selectedEngine
                const off =
                  (e.id === "sarvam" && !sarvamReady) ||
                  (e.id === "parler" && !parlerReady && selectedEngine !== "auto")
                return (
                  <button
                    key={e.id}
                    type="button"
                    className={`engine-chip ${on ? "on" : ""}`}
                    onClick={() => onEngineChange(e.id)}
                    title={e.hint}
                  >
                    {e.name}
                    {off && e.id !== "mms" && e.id !== "auto" ? " · off" : ""}
                  </button>
                )
              })}
            </div>

            <div className="voice-pop-title" style={{ marginTop: 12 }}>Speaking voice</div>
            <p className="voice-pop-note">{engineNote}</p>
            <div className="voice-list">
              {VOICES.map((v) => {
                const on = v.id === selectedVoice
                return (
                  <button
                    key={v.id}
                    type="button"
                    role="option"
                    aria-selected={on}
                    className={`voice-row ${on ? "on" : ""}`}
                    onClick={() => onVoiceChange(v.id)}
                  >
                    <span className="voice-avatar" style={{ background: `${v.tone}18`, color: v.tone }}>
                      {v.name.slice(0, 1)}
                    </span>
                    <span className="voice-meta">
                      <span className="voice-name">{v.name}</span>
                      <span className="voice-hint">{v.role} · {v.hint}</span>
                    </span>
                    {on && <Check size={16} color="var(--accent)" />}
                  </button>
                )
              })}
            </div>
            <button
              type="button"
              className="voice-try"
              disabled={disabled || isPreviewing}
              onClick={() => onPreview(selectedVoice)}
            >
              {isPreviewing ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Volume2 size={14} />}
              {isPreviewing ? "Playing sample" : `Try ${current.name}`}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
