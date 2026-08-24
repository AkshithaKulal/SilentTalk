import { motion, AnimatePresence } from "framer-motion"
import { Volume2, Loader2, Clock, TrendingUp } from "lucide-react"

const ConfBar = ({ pct, rank }) => {
  const color = pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#6366f1"
  return (
    <div style={{ height: 4, borderRadius: 4, background: "rgba(255,255,255,0.05)", overflow: "hidden", width: "100%" }}>
      <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }}
        transition={{ duration: 0.7, ease: "easeOut", delay: (rank || 0) * 0.05 }}
        style={{ height: "100%", borderRadius: 4, background: color }} />
    </div>
  )
}

export default function PredictionPanel({ prediction, translation, history, onSpeak, isSpeaking }) {
  const conf = prediction?.top_conf ?? 0
  const confColor = conf >= 70 ? "#22c55e" : conf >= 40 ? "#f59e0b" : "#ef4444"
  const confLabel = conf >= 70 ? "High confidence" : conf >= 40 ? "Medium" : "Low"
  const glowColor = conf >= 70 ? "rgba(34,197,94,0.15)" : conf >= 40 ? "rgba(245,158,11,0.12)" : "rgba(239,68,68,0.1)"

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 16, padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
          <TrendingUp size={13} style={{ color: "#6366f1" }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Prediction</span>
        </div>
        <AnimatePresence mode="wait">
          {!prediction ? (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, padding: "24px 0" }}>
              <div style={{ width: 56, height: 56, borderRadius: 14, border: "2px dashed rgba(99,102,241,0.2)",
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>🤟</div>
              <p style={{ fontSize: 12, color: "#334155", textAlign: "center", lineHeight: 1.5 }}>
                Perform a sign and capture<br />to see prediction here
              </p>
            </motion.div>
          ) : (
            <motion.div key={prediction.top_label} initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }}
              style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ textAlign: "center", padding: "20px 16px", borderRadius: 12,
                background: `radial-gradient(ellipse at center, ${glowColor}, transparent 70%)`,
                border: `1px solid ${confColor}30`, boxShadow: `0 0 30px ${glowColor}` }}>
                <motion.div initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
                  style={{ fontSize: 38, fontWeight: 900, color: "#f8fafc", letterSpacing: "-1px", marginBottom: 4 }}>
                  {prediction.top_label}
                </motion.div>
                <div style={{ fontSize: 13, fontWeight: 700, color: confColor }}>
                  {prediction.top_conf}% — {confLabel}
                </div>
              </div>
              {prediction.top5 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <p style={{ fontSize: 10, color: "#334155", textTransform: "uppercase", letterSpacing: "0.1em" }}>Top 5</p>
                  {prediction.top5.map((p, i) => (
                    <div key={p.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 10, color: "#334155", width: 12, flexShrink: 0 }}>{i + 1}</span>
                      <span style={{ fontSize: 12, color: i === 0 ? "#e2e8f0" : "#64748b",
                        flex: 1, fontWeight: i === 0 ? 700 : 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {p.label}
                      </span>
                      <div style={{ width: 80, flexShrink: 0 }}><ConfBar pct={p.conf} rank={i} /></div>
                      <span style={{ fontSize: 11, color: "#475569", width: 36, textAlign: "right", flexShrink: 0 }}>{p.conf}%</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {translation && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{ borderRadius: 16, overflow: "hidden", border: "1px solid rgba(129,140,248,0.25)",
              background: "linear-gradient(135deg, rgba(99,102,241,0.08), rgba(124,58,237,0.05))",
              boxShadow: "0 0 24px rgba(99,102,241,0.1)" }}>
            <div style={{ padding: "12px 16px 0", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <p style={{ fontSize: 10, color: "#6366f1", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 2 }}>Kannada Translation</p>
                <p style={{ fontSize: 9, color: "#334155" }}>checkpoint-1500 fine-tuned</p>
              </div>
              <motion.button onClick={onSpeak} disabled={isSpeaking}
                whileHover={!isSpeaking ? { scale: 1.05 } : {}} whileTap={!isSpeaking ? { scale: 0.95 } : {}}
                style={{ display: "flex", alignItems: "center", gap: 5, padding: "6px 12px", borderRadius: 8,
                  border: "1px solid rgba(129,140,248,0.3)", background: "rgba(99,102,241,0.1)",
                  color: "#a5b4fc", fontSize: 12, fontWeight: 600, cursor: isSpeaking ? "not-allowed" : "pointer" }}>
                {isSpeaking ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Volume2 size={12} />}
                {isSpeaking ? "Playing..." : "Speak"}
              </motion.button>
            </div>
            <div style={{ padding: "10px 16px 14px" }}>
              <p className="kannada" style={{ fontSize: 26, fontWeight: 800, color: "#c7d2fe", lineHeight: 1.4 }}>{translation}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div style={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 16, padding: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
          <Clock size={12} style={{ color: "#475569" }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Session History</span>
          {history.length > 0 && (
            <span style={{ marginLeft: "auto", fontSize: 11, color: "#334155",
              background: "rgba(255,255,255,0.04)", padding: "2px 8px", borderRadius: 12 }}>{history.length}</span>
          )}
        </div>
        {history.length === 0 ? (
          <p style={{ fontSize: 12, color: "#1e293b", textAlign: "center", padding: "12px 0" }}>No predictions yet</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 200, overflowY: "auto" }}>
            {history.map((h, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                  background: "rgba(255,255,255,0.02)", borderRadius: 8,
                  padding: "7px 10px", border: "1px solid rgba(255,255,255,0.04)" }}>
                <div style={{ minWidth: 0 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>{h.label}</span>
                  {h.translation && (
                    <span className="kannada" style={{ fontSize: 11, color: "#818cf8", marginLeft: 8 }}>{h.translation}</span>
                  )}
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{ fontSize: 11, color: "#475569" }}>{h.conf}%</div>
                  <div style={{ fontSize: 10, color: "#1e293b" }}>{h.time}</div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
