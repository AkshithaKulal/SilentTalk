import { useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Volume2, Loader2, Clock, TrendingUp, Plus, X, PlayCircle, Trash2, MessageSquare } from "lucide-react"

// ── Confidence bar used in top-5 list ────────────────────────────────────────
const ConfBar = ({ pct, rank }) => {
  const color = pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#6366f1"
  return (
    <div style={{ height: 4, borderRadius: 4, background: "rgba(255,255,255,0.05)", overflow: "hidden", width: "100%" }}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.7, ease: "easeOut", delay: (rank || 0) * 0.05 }}
        style={{ height: "100%", borderRadius: 4, background: color }}
      />
    </div>
  )
}

// ── Confidence colour helpers ─────────────────────────────────────────────────
const confColor = (c) => c >= 70 ? "#22c55e" : c >= 40 ? "#f59e0b" : "#ef4444"
const confLabel = (c) => c >= 70 ? "High confidence" : c >= 40 ? "Medium" : "Low"
const glowColor = (c) => c >= 70 ? "rgba(34,197,94,0.15)" : c >= 40 ? "rgba(245,158,11,0.12)" : "rgba(239,68,68,0.1)"

// ── Word chip in the sentence builder ────────────────────────────────────────
const WordChip = ({ word, conf, translation, onRemove }) => {
  const cc = confColor(conf)
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.7, y: -4 }}
      transition={{ duration: 0.18 }}
      style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        padding: "5px 8px 5px 10px", borderRadius: 20,
        background: "rgba(99,102,241,0.1)",
        border: `1px solid ${cc}40`,
        maxWidth: "100%",
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", whiteSpace: "nowrap" }}>{word}</div>
        <div className="kannada" style={{ fontSize: 11, color: "#818cf8", whiteSpace: "nowrap" }}>{translation}</div>
      </div>
      <button
        onClick={onRemove}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          width: 18, height: 18, borderRadius: "50%", border: "none",
          background: "rgba(255,255,255,0.06)", cursor: "pointer",
          color: "#475569", flexShrink: 0,
        }}
      >
        <X size={10} />
      </button>
    </motion.div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function PredictionPanel({
  prediction, translation, translating,
  sentence, history,
  isSpeaking, speakingTarget,
  onSpeakWord, onSpeakSentence,
  onAddToSentence, onRemoveFromSentence, onClearSentence,
}) {
  // Unique key per capture so AnimatePresence fires even for the same label
  const predCountRef = useRef(0)
  if (prediction) predCountRef.current += 1
  const animKey = prediction ? `${prediction.top_label}-${predCountRef.current}` : "empty"

  const conf  = prediction?.top_conf ?? 0
  const cc    = confColor(conf)
  const gc    = glowColor(conf)

  // Can we add this prediction to the sentence?
  const canAdd = !!prediction && !!translation && conf >= 60 && !translating

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* ── 1. Latest Prediction ─────────────────────────────────────────── */}
      <div style={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 16, padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
          <TrendingUp size={13} style={{ color: "#6366f1" }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Latest Prediction</span>
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
            <motion.div key={animKey} initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }}
              style={{ display: "flex", flexDirection: "column", gap: 10 }}>

              {/* Label + confidence */}
              <div style={{ textAlign: "center", padding: "16px 16px 12px", borderRadius: 12,
                background: `radial-gradient(ellipse at center, ${gc}, transparent 70%)`,
                border: `1px solid ${cc}30`, boxShadow: `0 0 30px ${gc}` }}>
                <motion.div initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
                  style={{ fontSize: 34, fontWeight: 900, color: "#f8fafc", letterSpacing: "-1px", marginBottom: 2 }}>
                  {prediction.top_label}
                </motion.div>
                <div style={{ fontSize: 12, fontWeight: 700, color: cc }}>
                  {prediction.top_conf}% — {confLabel(conf)}
                </div>
              </div>

              {/* Translation row */}
              <AnimatePresence>
                {translating && !translation && (
                  <motion.div key="tl-loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 10,
                      background: "rgba(99,102,241,0.05)", border: "1px solid rgba(99,102,241,0.1)" }}>
                    <Loader2 size={12} color="#818cf8" style={{ animation: "spin 1s linear infinite", flexShrink: 0 }} />
                    <span style={{ fontSize: 11, color: "#475569" }}>Translating...</span>
                  </motion.div>
                )}
                {translation && (
                  <motion.div key="tl-done" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    style={{ padding: "8px 12px", borderRadius: 10,
                      background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.15)" }}>
                    <div style={{ fontSize: 9, color: "#6366f1", textTransform: "uppercase",
                      letterSpacing: "0.1em", marginBottom: 3 }}>Kannada</div>
                    <p className="kannada" style={{ fontSize: 22, fontWeight: 800, color: "#c7d2fe", lineHeight: 1.3 }}>
                      {translation}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Action buttons row */}
              <div style={{ display: "flex", gap: 8 }}>
                {/* Speak this word */}
                <motion.button
                  onClick={onSpeakWord}
                  disabled={!translation || isSpeaking}
                  whileHover={translation && !isSpeaking ? { scale: 1.02 } : {}}
                  whileTap={translation && !isSpeaking ? { scale: 0.97 } : {}}
                  style={{
                    flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                    padding: "9px 0", borderRadius: 10, border: "1px solid rgba(129,140,248,0.25)",
                    background: "rgba(99,102,241,0.08)", color: "#a5b4fc",
                    fontSize: 12, fontWeight: 600,
                    cursor: !translation || isSpeaking ? "not-allowed" : "pointer",
                    opacity: !translation || isSpeaking ? 0.5 : 1,
                  }}
                >
                  {speakingTarget === 'word' && isSpeaking
                    ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} />
                    : <Volume2 size={12} />}
                  {speakingTarget === 'word' && isSpeaking ? "Playing..." : "Speak Word"}
                </motion.button>

                {/* Add to sentence */}
                <motion.button
                  onClick={() => canAdd && onAddToSentence(prediction.top_label, conf, translation)}
                  disabled={!canAdd}
                  whileHover={canAdd ? { scale: 1.02 } : {}}
                  whileTap={canAdd ? { scale: 0.97 } : {}}
                  title={conf < 60 ? `Confidence too low (${conf}%) — need ≥60%` : "Add this word to sentence"}
                  style={{
                    flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                    padding: "9px 0", borderRadius: 10,
                    background: canAdd ? "linear-gradient(135deg, #16a34a, #22c55e)" : "rgba(34,197,94,0.08)",
                    border: `1px solid ${canAdd ? "transparent" : "rgba(34,197,94,0.2)"}`,
                    color: canAdd ? "white" : "#4ade80",
                    fontSize: 12, fontWeight: 700,
                    cursor: canAdd ? "pointer" : "not-allowed",
                    opacity: canAdd ? 1 : 0.5,
                    boxShadow: canAdd ? "0 4px 14px rgba(34,197,94,0.25)" : "none",
                  }}
                >
                  <Plus size={13} />
                  Add to Sentence
                </motion.button>
              </div>

              {/* Low confidence warning */}
              {prediction && conf < 60 && (
                <div style={{ fontSize: 11, color: "#854d0e", background: "rgba(245,158,11,0.08)",
                  border: "1px solid rgba(245,158,11,0.2)", borderRadius: 8, padding: "6px 10px", textAlign: "center" }}>
                  ⚠ Confidence {conf}% — redo the sign for a cleaner capture before adding
                </div>
              )}

              {/* Top 5 */}
              {prediction.top5 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                  <p style={{ fontSize: 10, color: "#334155", textTransform: "uppercase", letterSpacing: "0.1em" }}>Top 5</p>
                  {prediction.top5.map((p, i) => (
                    <div key={p.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 10, color: "#334155", width: 12, flexShrink: 0 }}>{i + 1}</span>
                      <span style={{ fontSize: 12, color: i === 0 ? "#e2e8f0" : "#64748b",
                        flex: 1, fontWeight: i === 0 ? 700 : 400,
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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

      {/* ── 2. Sentence Builder ───────────────────────────────────────────── */}
      <div style={{
        background: "#0d1117", borderRadius: 16, padding: 16,
        border: sentence.length > 0
          ? "1px solid rgba(99,102,241,0.3)"
          : "1px solid rgba(255,255,255,0.06)",
        boxShadow: sentence.length > 0 ? "0 0 20px rgba(99,102,241,0.07)" : "none",
        transition: "all 0.3s",
      }}>
        {/* Header row */}
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12 }}>
          <MessageSquare size={13} style={{ color: "#6366f1" }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Sentence Builder</span>
          {sentence.length > 0 && (
            <span style={{ fontSize: 11, color: "#6366f1", background: "rgba(99,102,241,0.12)",
              padding: "1px 7px", borderRadius: 10, marginLeft: 2 }}>
              {sentence.length} word{sentence.length > 1 ? "s" : ""}
            </span>
          )}
          {sentence.length > 0 && (
            <button
              onClick={onClearSentence}
              style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4,
                fontSize: 11, color: "#475569", background: "none", border: "none",
                cursor: "pointer", padding: "3px 6px", borderRadius: 6 }}
            >
              <Trash2 size={11} /> Clear
            </button>
          )}
        </div>

        {/* Word chips */}
        {sentence.length === 0 ? (
          <div style={{ padding: "16px 0", textAlign: "center" }}>
            <p style={{ fontSize: 12, color: "#1e293b" }}>
              Add words using the button above
            </p>
            <p style={{ fontSize: 11, color: "#1e293b", marginTop: 4 }}>
              Only predictions ≥60% confidence can be added
            </p>
          </div>
        ) : (
          <>
            {/* Chips row */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12, minHeight: 40 }}>
              <AnimatePresence>
                {sentence.map((w) => (
                  <WordChip
                    key={w.id}
                    word={w.word}
                    conf={w.conf}
                    translation={w.translation}
                    onRemove={() => onRemoveFromSentence(w.id)}
                  />
                ))}
              </AnimatePresence>
            </div>

            {/* Full Kannada sentence preview */}
            <div style={{ padding: "10px 12px", borderRadius: 10, marginBottom: 12,
              background: "rgba(99,102,241,0.05)", border: "1px solid rgba(99,102,241,0.12)" }}>
              <div style={{ fontSize: 9, color: "#6366f1", textTransform: "uppercase",
                letterSpacing: "0.1em", marginBottom: 4 }}>Full sentence (Kannada)</div>
              <p className="kannada" style={{ fontSize: 18, fontWeight: 700, color: "#c7d2fe", lineHeight: 1.5 }}>
                {sentence.map(w => w.translation).join(' ')}
              </p>
            </div>

            {/* Speak sentence button */}
            <motion.button
              onClick={onSpeakSentence}
              disabled={isSpeaking}
              whileHover={!isSpeaking ? { scale: 1.01 } : {}}
              whileTap={!isSpeaking ? { scale: 0.98 } : {}}
              style={{
                width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                padding: "12px 0", borderRadius: 12, border: "none",
                background: isSpeaking
                  ? "rgba(99,102,241,0.15)"
                  : "linear-gradient(135deg, #4f46e5, #7c3aed)",
                color: isSpeaking ? "#818cf8" : "white",
                fontSize: 13, fontWeight: 700, cursor: isSpeaking ? "not-allowed" : "pointer",
                boxShadow: isSpeaking ? "none" : "0 4px 20px rgba(99,102,241,0.35)",
                transition: "all 0.2s",
              }}
            >
              {speakingTarget === 'sentence' && isSpeaking
                ? <Loader2 size={15} style={{ animation: "spin 1s linear infinite" }} />
                : <PlayCircle size={15} />}
              {speakingTarget === 'sentence' && isSpeaking
                ? "Speaking sentence..."
                : `Speak Sentence (${sentence.length} word${sentence.length > 1 ? "s" : ""})`}
            </motion.button>
          </>
        )}
      </div>

      {/* ── 3. Session History ────────────────────────────────────────────── */}
      <div style={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 16, padding: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
          <Clock size={12} style={{ color: "#475569" }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>Session History</span>
          {history.length > 0 && (
            <span style={{ marginLeft: "auto", fontSize: 11, color: "#334155",
              background: "rgba(255,255,255,0.04)", padding: "2px 8px", borderRadius: 12 }}>
              {history.length}
            </span>
          )}
        </div>

        {history.length === 0 ? (
          <p style={{ fontSize: 12, color: "#1e293b", textAlign: "center", padding: "12px 0" }}>
            Spoken sentences will appear here
          </p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 220, overflowY: "auto" }}>
            <AnimatePresence>
              {history.map((h, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  style={{ background: "rgba(255,255,255,0.02)", borderRadius: 10,
                    padding: "9px 12px", border: "1px solid rgba(255,255,255,0.04)" }}>
                  {/* English words */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 3 }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>{h.sentence}</span>
                    <span style={{ fontSize: 10, color: "#334155", flexShrink: 0, marginLeft: 8 }}>{h.time}</span>
                  </div>
                  {/* Kannada */}
                  <p className="kannada" style={{ fontSize: 13, color: "#818cf8" }}>{h.kannada}</p>
                  <div style={{ fontSize: 10, color: "#1e293b", marginTop: 2 }}>
                    {h.wordCount} word{h.wordCount > 1 ? "s" : ""} spoken
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

    </div>
  )
}
