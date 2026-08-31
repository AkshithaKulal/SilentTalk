import { Hand, BookOpen } from "lucide-react"

export default function Header({ status, libraryOpen, onToggleLibrary }) {
  const ready = status.classifier && status.translation_model && status.tts_model

  return (
    <header
      style={{
        background: "var(--surface)",
        borderBottom: "1px solid var(--line)",
        position: "sticky",
        top: 0,
        zIndex: 40,
      }}
    >
      <div
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "0 20px",
          height: 56,
          display: "flex",
          alignItems: "center",
          gap: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: "var(--accent-soft)",
              color: "var(--accent)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Hand size={16} strokeWidth={2.4} />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.02em", lineHeight: 1.1 }}>
              SilentTalk
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Sign to Kannada speech</div>
          </div>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              fontWeight: 600,
              color: ready ? "var(--ok)" : "var(--muted)",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: ready ? "var(--ok)" : "var(--faint)",
              }}
            />
            {ready ? "Ready" : "Starting"}
          </span>

          <button
            type="button"
            onClick={onToggleLibrary}
            aria-pressed={libraryOpen}
            style={{
              height: 34,
              padding: "0 12px",
              borderRadius: 8,
              border: "1px solid var(--line)",
              background: libraryOpen ? "var(--accent-soft)" : "var(--surface)",
              color: libraryOpen ? "var(--accent-ink)" : "var(--ink)",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <BookOpen size={14} />
            Practice signs
          </button>
        </div>
      </div>
    </header>
  )
}
