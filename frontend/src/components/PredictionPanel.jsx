import { useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Volume2, Loader2, Clock, TrendingUp, X, PlayCircle, Trash2, MessageSquare, RotateCcw } from "lucide-react"

// ── Confidence helpers ────────────────────────────────────────────────────────
const cc  = (c) => c >= 70 ? "#059669" : c >= 40 ? "#d97706" : "#dc2626"
const ccBg = (c) => c >= 70 ? "rgba(5,150,105,0.08)"  : c >= 40 ? "rgba(217,119,6,0.08)"  : "rgba(220,38,38,0.08)"
const ccBorder = (c) => c >= 70 ? "rgba(5,150,105,0.2)" : c >= 40 ? "rgba(217,119,6,0.2)" : "rgba(220,38,38,0.2)"
const cl  = (c) => c >= 70 ? "High confidence" : c >= 40 ? "Medium" : "Low"

// ── Confidence bar ────────────────────────────────────────────────────────────
const ConfBar = ({ pct, rank }) => (
  <div style={{ height: 5, borderRadius: 4, background: '#f1f5f9', overflow: 'hidden', width: '100%' }}>
    <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }}
      transition={{ duration: 0.6, ease: 'easeOut', delay: (rank || 0) * 0.04 }}
      style={{ height: '100%', borderRadius: 4,
        background: pct >= 70 ? '#059669' : pct >= 40 ? '#d97706' : '#dc2626' }} />
  </div>
)

// ── Word chip ─────────────────────────────────────────────────────────────────
const WordChip = ({ word, conf, translation, onRemove }) => (
  <motion.div initial={{ opacity: 0, scale: 0.8, y: 4 }} animate={{ opacity: 1, scale: 1, y: 0 }}
    exit={{ opacity: 0, scale: 0.7 }} transition={{ duration: 0.15 }}
    style={{ display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '6px 8px 6px 10px', borderRadius: 10,
      background: '#f8fafc', border: '1px solid #e2e8f0',
      boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#0f172a', whiteSpace: 'nowrap' }}>{word}</div>
      <div className="kannada" style={{ fontSize: 11, color: '#0d9488', whiteSpace: 'nowrap' }}>{translation}</div>
    </div>
    <button onClick={onRemove}
      style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: 18, height: 18, borderRadius: '50%', border: '1px solid #e2e8f0',
        background: '#fff', cursor: 'pointer', color: '#94a3b8', flexShrink: 0 }}>
      <X size={9} />
    </button>
  </motion.div>
)

// ── Main ──────────────────────────────────────────────────────────────────────
export default function PredictionPanel({
  prediction, translation, translating,
  sentence, history,
  isSpeaking, speakingTarget,
  voices, selectedVoice, onVoiceChange,
  onSpeakWord, onSpeakSentence, onReplayHistory,
  onRemoveFromSentence, onClearSentence,
}) {
  const predCountRef = useRef(0)
  if (prediction) predCountRef.current += 1
  const animKey = prediction ? `${prediction.top_label}-${predCountRef.current}` : "empty"

  const conf   = prediction?.top_conf ?? 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* ── Voice Switcher ─────────────────────────────────────────────────── */}
      {voices && voices.length > 0 && (
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 14,
          padding: '10px 14px', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <Volume2 size={13} color="#0d9488" />
            <span style={{ fontSize: 12, fontWeight: 700, color: '#0f172a' }}>Voice</span>
            <span style={{ fontSize: 10, color: '#94a3b8', marginLeft: 2 }}>Kannada TTS</span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {voices.map(v => (
              <button key={v.id} onClick={() => onVoiceChange(v.id)}
                title={v.description}
                style={{
                  padding: '5px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                  cursor: 'pointer', transition: 'all 0.15s', border: 'none',
                  background: selectedVoice === v.id
                    ? 'linear-gradient(135deg, #0d9488, #059669)'
                    : '#f1f5f9',
                  color: selectedVoice === v.id ? 'white' : '#64748b',
                  boxShadow: selectedVoice === v.id ? '0 2px 8px rgba(13,148,136,0.25)' : 'none',
                }}>
                {v.name}
              </button>
            ))}
          </div>
          {voices.find(v => v.id === selectedVoice) && (
            <p style={{ fontSize: 10, color: '#94a3b8', marginTop: 6 }}>
              {voices.find(v => v.id === selectedVoice).description}
            </p>
          )}
        </div>
      )}

      {/* ── 1. Latest Prediction ──────────────────────────────────────────── */}
      <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 16,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)', overflow: 'hidden' }}>

        {/* Card header */}
        <div style={{ padding: '12px 14px', borderBottom: '1px solid #f1f5f9', background: '#fafafa',
          display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: 'rgba(13,148,136,0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <TrendingUp size={13} color="#0d9488" />
          </div>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>Latest Prediction</span>
        </div>

        <div style={{ padding: 14 }}>
          <AnimatePresence mode="wait">
            {!prediction ? (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
                  gap: 10, padding: '24px 0' }}>
                <div style={{ width: 60, height: 60, borderRadius: 16,
                  background: 'rgba(13,148,136,0.06)', border: '1.5px dashed rgba(13,148,136,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 26 }}>🤟</div>
                <p style={{ fontSize: 13, color: '#94a3b8', textAlign: 'center', lineHeight: 1.5 }}>
                  Perform a sign in Go Live<br />High-confidence words join the sentence automatically
                </p>
              </motion.div>
            ) : (
              <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

                {/* Prediction hero */}
                <div style={{ padding: '16px 14px 12px', borderRadius: 12,
                  background: ccBg(conf), border: `1px solid ${ccBorder(conf)}` }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                    <div>
                      <motion.div initial={{ y: 6, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
                        style={{ fontSize: 32, fontWeight: 900, color: '#0f172a',
                          letterSpacing: '-0.5px', lineHeight: 1.1 }}>
                        {prediction.top_label}
                      </motion.div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: cc(conf), marginTop: 4 }}>
                        {prediction.top_conf}% — {cl(conf)}
                      </div>
                    </div>
                    {/* Circular confidence ring */}
                    <div style={{ flexShrink: 0, width: 48, height: 48, borderRadius: '50%',
                      background: '#fff', border: `3px solid ${cc(conf)}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      boxShadow: `0 2px 8px ${ccBg(conf)}` }}>
                      <span style={{ fontSize: 11, fontWeight: 800, color: cc(conf) }}>{conf}%</span>
                    </div>
                  </div>
                </div>

                {/* Kannada translation */}
                <AnimatePresence>
                  {translating && (
                    <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px',
                        borderRadius: 10, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                      <Loader2 size={13} color="#0d9488" style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                      <span style={{ fontSize: 12, color: '#64748b', fontWeight: 500 }}>Translating to Kannada...</span>
                    </motion.div>
                  )}
                  {translation && !translating && (
                    <motion.div key="done" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                      style={{ padding: '10px 12px', borderRadius: 10,
                        background: 'rgba(13,148,136,0.04)', border: '1px solid rgba(13,148,136,0.15)' }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: '#0d9488',
                        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
                        Kannada Translation
                      </div>
                      <p className="kannada" style={{ fontSize: 24, fontWeight: 800,
                        color: '#0f172a', lineHeight: 1.3 }}>
                        {translation}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                  <motion.button onClick={onSpeakWord}
                    disabled={!prediction || isSpeaking}
                    whileHover={prediction && !isSpeaking ? { scale: 1.02 } : {}}
                    whileTap={prediction && !isSpeaking ? { scale: 0.97 } : {}}
                    style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                      padding: '9px 0', borderRadius: 10,
                      background: '#f8fafc', border: '1px solid #e2e8f0',
                      color: !prediction || isSpeaking ? '#94a3b8' : '#334155',
                      fontSize: 12, fontWeight: 600,
                      cursor: !prediction || isSpeaking ? 'not-allowed' : 'pointer',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                    {speakingTarget === 'word' && isSpeaking
                      ? <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} />
                      : <Volume2 size={12} />}
                    {speakingTarget === 'word' && isSpeaking ? "Playing..." : "Preview this word"}
                  </motion.button>

                {/* Top 5 */}
                {prediction.top5 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6,
                    padding: '10px 0 0', borderTop: '1px solid #f1f5f9' }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8',
                      textTransform: 'uppercase', letterSpacing: '0.08em' }}>Top 5</p>
                    {prediction.top5.map((p, i) => (
                      <div key={p.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 10, color: '#94a3b8', width: 12, flexShrink: 0,
                          fontWeight: 600 }}>{i + 1}</span>
                        <span style={{ fontSize: 12, flex: 1,
                          color: i === 0 ? '#0f172a' : '#64748b',
                          fontWeight: i === 0 ? 700 : 500,
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {p.label}
                        </span>
                        <div style={{ width: 72, flexShrink: 0 }}><ConfBar pct={p.conf} rank={i} /></div>
                        <span style={{ fontSize: 11, color: '#64748b', width: 34,
                          textAlign: 'right', flexShrink: 0, fontWeight: 600 }}>{p.conf}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── 2. Sentence Builder ────────────────────────────────────────────── */}
      <div style={{
        background: '#fff', borderRadius: 16, overflow: 'hidden',
        border: sentence.length > 0 ? '1.5px solid rgba(13,148,136,0.25)' : '1px solid #e2e8f0',
        boxShadow: sentence.length > 0 ? '0 4px 16px rgba(13,148,136,0.08)' : '0 1px 4px rgba(0,0,0,0.06)',
        transition: 'all 0.3s',
      }}>
        {/* Header */}
        <div style={{ padding: '12px 14px', borderBottom: '1px solid #f1f5f9', background: '#fafafa',
          display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: 'rgba(13,148,136,0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <MessageSquare size={13} color="#0d9488" />
          </div>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>Sentence Builder</span>
          {sentence.length > 0 && (
            <span style={{ fontSize: 11, fontWeight: 700, color: '#0d9488',
              background: 'rgba(13,148,136,0.08)', padding: '2px 8px', borderRadius: 10,
              border: '1px solid rgba(13,148,136,0.2)' }}>
              {sentence.length} word{sentence.length > 1 ? 's' : ''}
            </span>
          )}
          {/* Always show Clear — disabled when empty */}
          <button onClick={onClearSentence}
            disabled={sentence.length === 0}
            style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: 11, fontWeight: 600,
              color: sentence.length > 0 ? '#dc2626' : '#cbd5e1',
              background: sentence.length > 0 ? 'rgba(220,38,38,0.06)' : 'transparent',
              border: `1px solid ${sentence.length > 0 ? 'rgba(220,38,38,0.2)' : '#e2e8f0'}`,
              cursor: sentence.length > 0 ? 'pointer' : 'not-allowed',
              padding: '3px 8px', borderRadius: 8, transition: 'all 0.15s' }}>
            <Trash2 size={11} /> Clear
          </button>
        </div>

        <div style={{ padding: 14 }}>
          {sentence.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <div style={{ fontSize: 24, marginBottom: 6 }}>📝</div>
              <p style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8' }}>No words yet</p>
              <p style={{ fontSize: 11, color: '#cbd5e1', marginTop: 4 }}>
                Go Live and sign. Words append on their own. Press Speak when ready.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {/* Word chips */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                <AnimatePresence>
                  {sentence.map(w => (
                    <WordChip key={w.id} word={w.word} conf={w.conf}
                      translation={w.translation} onRemove={() => onRemoveFromSentence(w.id)} />
                  ))}
                </AnimatePresence>
              </div>

              {/* Kannada preview */}
              <div style={{ padding: '10px 12px', borderRadius: 10,
                background: 'rgba(13,148,136,0.04)', border: '1px solid rgba(13,148,136,0.12)' }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: '#0d9488',
                  textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
                  Full sentence · Kannada
                </div>
                <p className="kannada" style={{ fontSize: 20, fontWeight: 700,
                  color: '#0f172a', lineHeight: 1.5 }}>
                  {sentence.map(w => w.translation).join(' ')}
                </p>
              </div>

              {/* Speak button */}
              <motion.button onClick={onSpeakSentence} disabled={isSpeaking}
                whileHover={!isSpeaking ? { scale: 1.01 } : {}}
                whileTap={!isSpeaking ? { scale: 0.98 } : {}}
                style={{ width: '100%', display: 'flex', alignItems: 'center',
                  justifyContent: 'center', gap: 8, padding: '13px 0', borderRadius: 12,
                  background: isSpeaking ? '#f8fafc' : 'linear-gradient(135deg, #0d9488, #059669)',
                  border: isSpeaking ? '1px solid #e2e8f0' : 'none',
                  color: isSpeaking ? '#94a3b8' : 'white',
                  fontSize: 13, fontWeight: 700, cursor: isSpeaking ? 'not-allowed' : 'pointer',
                  boxShadow: isSpeaking ? 'none' : '0 4px 16px rgba(13,148,136,0.3)',
                  transition: 'all 0.2s' }}>
                {speakingTarget === 'sentence' && isSpeaking
                  ? <Loader2 size={15} style={{ animation: 'spin 1s linear infinite' }} />
                  : <PlayCircle size={15} />}
                {speakingTarget === 'sentence' && isSpeaking
                  ? 'Speaking...'
                  : `Speak Sentence (${sentence.length} word${sentence.length > 1 ? 's' : ''})`}
              </motion.button>
            </div>
          )}
        </div>
      </div>

      {/* ── 3. Session History ─────────────────────────────────────────────── */}
      <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 16,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)', overflow: 'hidden' }}>
        <div style={{ padding: '12px 14px', borderBottom: '1px solid #f1f5f9', background: '#fafafa',
          display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: '#f1f5f9',
            display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Clock size={13} color="#64748b" />
          </div>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>Session History</span>
          {history.length > 0 && (
            <span style={{ marginLeft: 'auto', fontSize: 11, fontWeight: 600, color: '#64748b',
              background: '#f1f5f9', padding: '2px 8px', borderRadius: 10 }}>
              {history.length}
            </span>
          )}
        </div>

        <div style={{ padding: 14 }}>
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
              <div style={{ fontSize: 22, marginBottom: 6 }}>📋</div>
              <p style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500 }}>
                Spoken sentences will appear here
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6,
              maxHeight: 220, overflowY: 'auto' }}>
              <AnimatePresence>
                {history.map((h, i) => {
                  const isReplaying = speakingTarget === 'replay' && isSpeaking && i === 0
                  return (
                    <motion.div key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
                      style={{ background: '#f8fafc', borderRadius: 10, padding: '9px 12px',
                        border: isReplaying ? '1px solid rgba(13,148,136,0.3)' : '1px solid #f1f5f9',
                        transition: 'border 0.2s' }}>

                      {/* Top row: English + time + replay button */}
                      <div style={{ display: 'flex', justifyContent: 'space-between',
                        alignItems: 'flex-start', gap: 8, marginBottom: 4 }}>
                        <div style={{ minWidth: 0, flex: 1 }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: '#0f172a' }}>
                            {h.sentence}
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                          <span style={{ fontSize: 10, color: '#94a3b8' }}>{h.time}</span>
                          {/* Replay button */}
                          <motion.button
                            onClick={() => onReplayHistory(h.kannada)}
                            disabled={isSpeaking}
                            whileHover={!isSpeaking ? { scale: 1.1 } : {}}
                            whileTap={!isSpeaking ? { scale: 0.9 } : {}}
                            title="Replay this sentence"
                            style={{
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              width: 26, height: 26, borderRadius: 8,
                              background: isReplaying
                                ? 'rgba(13,148,136,0.15)'
                                : 'rgba(13,148,136,0.08)',
                              border: `1px solid ${isReplaying
                                ? 'rgba(13,148,136,0.4)'
                                : 'rgba(13,148,136,0.2)'}`,
                              cursor: isSpeaking ? 'not-allowed' : 'pointer',
                              opacity: isSpeaking && !isReplaying ? 0.4 : 1,
                              transition: 'all 0.15s',
                            }}>
                            {isReplaying
                              ? <Loader2 size={11} color="#0d9488" style={{ animation: 'spin 1s linear infinite' }} />
                              : <RotateCcw size={11} color="#0d9488" />}
                          </motion.button>
                        </div>
                      </div>

                      {/* Kannada text */}
                      <p className="kannada" style={{ fontSize: 14, fontWeight: 600, color: '#0d9488' }}>
                        {h.kannada}
                      </p>
                      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>
                        {h.wordCount} word{h.wordCount > 1 ? 's' : ''} spoken
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
