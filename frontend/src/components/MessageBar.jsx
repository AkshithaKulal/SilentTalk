import { Volume2, Trash2, X, Loader2 } from "lucide-react"

/**
 * AAC-style message window (Proloquo2Go / TD Snap pattern):
 * the composed utterance sits at the top, large and always visible.
 * Speak is the only primary action. Words are removable chips.
 */
export default function MessageBar({
  sentence,
  isSpeaking,
  onSpeak,
  onClear,
  onRemove,
}) {
  const english = sentence.map((w) => w.word).join("  ·  ")
  const kannada = sentence.map((w) => w.translation).filter(Boolean).join(" ")
  const canSpeak = sentence.length > 0 && !isSpeaking

  return (
    <section
      style={{
        maxWidth: 1280,
        margin: "0 auto",
        width: "100%",
        padding: "12px 20px 16px",
      }}
    >
      <div className="message-card">
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              color: "var(--faint)",
              marginBottom: 8,
            }}
          >
            Message
          </div>

          {sentence.length === 0 ? (
            <p style={{ fontSize: 22, fontWeight: 600, color: "var(--faint)", lineHeight: 1.35 }}>
              Sign in front of the camera. Words appear here.
            </p>
          ) : (
            <>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
                {sentence.map((w) => (
                  <span
                    key={w.id}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "6px 8px 6px 12px",
                      borderRadius: 999,
                      background: "var(--accent-soft)",
                      color: "var(--accent-ink)",
                      fontSize: 14,
                      fontWeight: 600,
                    }}
                  >
                    {w.word}
                    <button
                      type="button"
                      onClick={() => onRemove(w.id)}
                      aria-label={`Remove ${w.word}`}
                      style={{
                        width: 20,
                        height: 20,
                        border: "none",
                        borderRadius: "50%",
                        background: "transparent",
                        color: "var(--accent)",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
              <p
                className="kannada"
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "var(--ink)",
                  lineHeight: 1.4,
                  minHeight: 40,
                }}
              >
                {kannada || english}
              </p>
            </>
          )}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8, justifyContent: "center" }}>
          <button
            type="button"
            className="btn-primary"
            onClick={onSpeak}
            disabled={!canSpeak}
            style={{
              minWidth: 148,
              height: 52,
              borderRadius: 12,
              border: "none",
              cursor: canSpeak ? "pointer" : "not-allowed",
              background: canSpeak ? "var(--accent)" : "var(--line)",
              color: canSpeak ? "#fff" : "var(--faint)",
              fontSize: 16,
              fontWeight: 700,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
            }}
          >
            {isSpeaking ? <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> : <Volume2 size={18} />}
            {isSpeaking ? "Speaking" : "Speak"}
          </button>
          <button
            type="button"
            onClick={onClear}
            disabled={sentence.length === 0}
            style={{
              height: 36,
              borderRadius: 10,
              border: "1px solid var(--line)",
              background: "var(--surface)",
              color: sentence.length ? "var(--muted)" : "var(--faint)",
              cursor: sentence.length ? "pointer" : "not-allowed",
              fontSize: 13,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
            }}
          >
            <Trash2 size={13} />
            Clear
          </button>
        </div>
      </div>
    </section>
  )
}
