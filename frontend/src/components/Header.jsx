import { motion } from "framer-motion"
import { BookOpen, Hand } from "lucide-react"
import VoicePicker from "./VoicePicker"

export default function Header({
  status,
  libraryOpen,
  onToggleLibrary,
  selectedVoice,
  selectedEngine,
  onVoiceChange,
  onEngineChange,
  onPreviewVoice,
  isPreviewing,
  parlerReady,
  sarvamReady,
  voiceBusy,
}) {
  const ready = status.classifier && status.translation_model && status.tts_model

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 40,
        backdropFilter: "blur(16px)",
        background: "rgba(255,255,255,0.78)",
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div
        style={{
          maxWidth: 1360,
          margin: "0 auto",
          padding: "0 20px",
          height: 56,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ display: "flex", alignItems: "center", gap: 10 }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 11,
              background: "linear-gradient(145deg, #3b82f6, #1d4ed8)",
              color: "#fff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 8px 18px rgba(37,99,235,0.28)",
            }}
          >
            <Hand size={16} strokeWidth={2.4} />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
              SilentTalk
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Sign · Kannada · Speak</div>
          </div>
        </motion.div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              fontWeight: 600,
              color: ready ? "var(--ok)" : "var(--warn)",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: ready ? "var(--ok)" : "var(--warn)",
                boxShadow: ready ? "0 0 0 4px rgba(5,150,105,0.15)" : "0 0 0 4px rgba(217,119,6,0.15)",
              }}
            />
            {ready ? "Ready" : "UI only"}
          </span>

          <VoicePicker
            selectedVoice={selectedVoice}
            selectedEngine={selectedEngine}
            onVoiceChange={onVoiceChange}
            onEngineChange={onEngineChange}
            onPreview={onPreviewVoice}
            isPreviewing={isPreviewing}
            parlerReady={parlerReady}
            sarvamReady={sarvamReady}
            disabled={voiceBusy}
          />

          <button
            type="button"
            onClick={onToggleLibrary}
            aria-pressed={libraryOpen}
            className="header-btn"
            style={{
              background: libraryOpen ? "var(--accent-soft)" : "var(--surface)",
              color: libraryOpen ? "var(--accent-ink)" : "var(--ink)",
            }}
          >
            <BookOpen size={14} />
            Practice
          </button>
        </div>
      </div>
    </header>
  )
}
