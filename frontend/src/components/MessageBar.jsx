import { AnimatePresence, motion } from "framer-motion"
import { Volume2, Trash2, X, Undo2 } from "lucide-react"

export default function MessageBar({
  sentence,
  isSpeaking,
  speakingName,
  onSpeak,
  onClear,
  onRemove,
  onUndo,
  canSpeak = false,
  speakHint = "Sign first, then tap Speak",
}) {
  const english = sentence.map((w) => w.word).join(" · ")
  const kannada = sentence.map((w) => w.translation).filter(Boolean).join(" ")
  const empty = sentence.length === 0
  const ready = canSpeak && !isSpeaking

  return (
    <section className="message-dock">
      <motion.div
        className="message-card message-card--speak"
        layout
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 320, damping: 26 }}
      >
        <div style={{ minWidth: 0, flex: 1 }}>
          {empty ? (
            <p style={{ fontSize: 14, color: "var(--muted)", fontWeight: 600, marginBottom: 8 }}>
              {speakHint}
            </p>
          ) : (
            <>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                <AnimatePresence initial={false}>
                  {sentence.map((w) => (
                    <motion.span
                      key={w.id}
                      className="chip"
                      layout
                      initial={{ opacity: 0, scale: 0.7, y: 8 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ type: "spring", stiffness: 500, damping: 28 }}
                    >
                      {w.word}
                      <button
                        type="button"
                        onClick={() => onRemove(w.id)}
                        aria-label={`Remove ${w.word}`}
                        style={{
                          width: 20, height: 20, border: "none", borderRadius: "50%",
                          background: "transparent", color: "var(--accent)", cursor: "pointer",
                          display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                      >
                        <X size={12} />
                      </button>
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
              <motion.p
                key={kannada || english}
                className="kannada"
                initial={{ opacity: 0.4, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ fontSize: 22, fontWeight: 800, lineHeight: 1.3 }}
              >
                {kannada || english}
              </motion.p>
              {kannada && english && (
                <p style={{ marginTop: 2, fontSize: 12, color: "var(--muted)" }}>{english}</p>
              )}
            </>
          )}
        </div>

        <div className="speak-col">
          <motion.button
            type="button"
            className="btn-primary speak-main-btn"
            onClick={onSpeak}
            disabled={!ready}
            whileTap={ready ? { scale: 0.97 } : {}}
            style={{
              width: "100%",
              minWidth: 120,
              height: 52,
              borderRadius: 14,
              border: "none",
              cursor: ready ? "pointer" : "not-allowed",
              background: ready ? "var(--accent)" : "var(--line)",
              color: ready ? "#fff" : "var(--faint)",
              fontSize: 17,
              fontWeight: 800,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
              boxShadow: ready ? "0 12px 24px rgba(37,99,235,0.28)" : "none",
            }}
          >
            {isSpeaking ? (
              <>
                <span className="eq" aria-hidden="true"><i /><i /><i /><i /></span>
                {speakingName || "Speaking"}
              </>
            ) : (
              <><Volume2 size={20} /> Speak</>
            )}
          </motion.button>
          {!empty && (
            <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
              <button type="button" onClick={onUndo} style={dockBtn}>
                <Undo2 size={13} /> Undo
              </button>
              <button type="button" onClick={onClear} style={dockBtn}>
                <Trash2 size={13} /> Clear
              </button>
            </div>
          )}
        </div>
      </motion.div>
    </section>
  )
}

const dockBtn = {
  flex: 1, height: 34, borderRadius: 10, border: "1px solid var(--line)",
  background: "var(--surface)", color: "var(--muted)", cursor: "pointer",
  fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
}
