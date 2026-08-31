import { AnimatePresence, motion } from "framer-motion"
import { Loader2, RotateCcw } from "lucide-react"

export default function LiveRail({
  history,
  isSpeaking,
  speakingTarget,
  onReplayHistory,
}) {
  return (
    <aside className="live-rail" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
      <motion.section
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: 22,
          padding: 16,
          boxShadow: "var(--shadow)",
        }}
      >
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--faint)", marginBottom: 10 }}>
          Spoken
        </div>
        {history.length === 0 ? (
          <>
            <div className="wave-empty" aria-hidden="true">
              <span /><span /><span /><span /><span /><span /><span />
            </div>
            <p style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Nothing spoken yet</p>
            <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.55 }}>
              After you Speak, Kannada audio stays here so someone nearby can hear it again.
            </p>
          </>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, minHeight: 0, overflowY: "auto" }}>
            <AnimatePresence initial={false}>
              {history.map((h, i) => (
                <motion.div
                  key={`${h.time}-${i}`}
                  layout
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 14,
                    background: "var(--bg)",
                    display: "flex",
                    gap: 8,
                    alignItems: "flex-start",
                  }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{h.sentence}</div>
                    <div className="kannada" style={{ fontSize: 15, color: "var(--accent-ink)", marginTop: 2 }}>
                      {h.kannada}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 4 }}>{h.time}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onReplayHistory(h.kannada)}
                    disabled={isSpeaking}
                    aria-label="Replay"
                    style={{
                      width: 32, height: 32, borderRadius: 10, border: "1px solid var(--line)",
                      background: "var(--surface)", cursor: isSpeaking ? "not-allowed" : "pointer",
                      color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center",
                      flexShrink: 0,
                    }}
                  >
                    {speakingTarget === "replay" && isSpeaking && i === 0 ? (
                      <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                    ) : (
                      <RotateCcw size={14} />
                    )}
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </motion.section>
    </aside>
  )
}
