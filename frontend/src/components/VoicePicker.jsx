import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { Check, Loader2, Volume2 } from "lucide-react"
import { VOICES, voiceById } from "../voices"

export default function VoicePicker({
  selectedVoice,
  onVoiceChange,
  onPreview,
  isPreviewing,
  parlerReady,
  disabled,
}) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef(null)
  const current = voiceById(selectedVoice)

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

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="listbox"
        className="header-btn"
        title="Speaking voice"
      >
        <span className="voice-dot" style={{ background: current.tone }} />
        {current.name}
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
            <div className="voice-pop-title">Speaking voice</div>
            <p className="voice-pop-note">
              {parlerReady
                ? "Speak uses this voice. Try a sample before you sign."
                : "Quality voices need the speech model on the server. Until then Speak uses a single Kannada voice."}
            </p>
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
