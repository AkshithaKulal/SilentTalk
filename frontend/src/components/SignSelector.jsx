import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, ChevronRight, BookOpen } from 'lucide-react'

// Assign category color based on sign label
const getCategoryColor = (label) => {
  const l = label.toLowerCase()
  if (['hello','thank you','please','sorry','good morning','how are you','bye'].includes(l))
    return { bg: 'rgba(99,102,241,0.12)', border: 'rgba(99,102,241,0.3)', text: '#a5b4fc', glow: 'rgba(99,102,241,0.2)' }
  if (['i','you','he','she','we','they'].includes(l))
    return { bg: 'rgba(20,184,166,0.1)', border: 'rgba(20,184,166,0.3)', text: '#5eead4', glow: 'rgba(20,184,166,0.2)' }
  if (['mother','father','brother','sister','friend'].includes(l))
    return { bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.3)', text: '#fcd34d', glow: 'rgba(245,158,11,0.2)' }
  if (['house','school','hospital','park','restaurant','market'].includes(l))
    return { bg: 'rgba(34,197,94,0.08)', border: 'rgba(34,197,94,0.25)', text: '#86efac', glow: 'rgba(34,197,94,0.15)' }
  if (['monday','tuesday','wednesday','thursday','friday','saturday','sunday','today','tomorrow','yesterday'].includes(l))
    return { bg: 'rgba(244,114,182,0.08)', border: 'rgba(244,114,182,0.25)', text: '#f9a8d4', glow: 'rgba(244,114,182,0.15)' }
  return { bg: 'rgba(99,102,241,0.08)', border: 'rgba(99,102,241,0.2)', text: '#c7d2fe', glow: 'rgba(99,102,241,0.1)' }
}

export default function SignSelector({ onSelect, selected }) {
  const [signs, setSigns] = useState([])

  useEffect(() => {
    fetch('/api/signs').then(r => r.json()).then(setSigns).catch(() => setSigns([]))
  }, [])

  return (
    <div className="flex flex-col gap-4">
      {/* Sign grid card */}
      <div style={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 16, padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BookOpen size={14} style={{ color: '#6366f1' }} />
            <span style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>Choose a Sign</span>
          </div>
          <span style={{ fontSize: 11, color: '#475569', background: 'rgba(255,255,255,0.04)', padding: '2px 8px', borderRadius: 20, border: '1px solid rgba(255,255,255,0.06)' }}>
            {signs.length} signs
          </span>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 220, overflowY: 'auto', paddingRight: 2 }}>
          {signs.map(sign => {
            const c = getCategoryColor(sign.label)
            const isActive = selected?.label === sign.label
            return (
              <motion.button
                key={sign.label}
                onClick={() => onSelect(sign)}
                whileHover={{ scale: 1.04, y: -1 }}
                whileTap={{ scale: 0.96 }}
                style={{
                  padding: '6px 12px',
                  borderRadius: 8,
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: `1px solid ${isActive ? c.border : 'rgba(255,255,255,0.06)'}`,
                  background: isActive ? c.bg : 'rgba(255,255,255,0.02)',
                  color: isActive ? c.text : '#64748b',
                  boxShadow: isActive ? `0 0 12px ${c.glow}` : 'none',
                  transition: 'all 0.15s ease',
                  whiteSpace: 'nowrap',
                }}
              >
                {sign.label}
              </motion.button>
            )
          })}
          {signs.length === 0 && (
            <p style={{ color: '#334155', fontSize: 12, padding: 8 }}>No signs found — check verification_set/</p>
          )}
        </div>
      </div>

      {/* Reference video */}
      <AnimatePresence>
        {selected && (() => {
          const c = getCategoryColor(selected.label)
          return (
            <motion.div
              key={selected.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              style={{
                borderRadius: 16,
                border: `1px solid ${c.border}`,
                background: '#0d1117',
                overflow: 'hidden',
                boxShadow: `0 0 24px ${c.glow}`,
              }}
            >
              <div style={{ padding: '12px 16px 8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 2 }}>Reference Sign</div>
                  <div style={{ fontSize: 18, fontWeight: 800, color: c.text }}>{selected.label}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: c.text, opacity: 0.7 }}>
                  <Play size={11} fill="currentColor" />
                  Watch & imitate
                </div>
              </div>

              <video
                key={selected.label}
                src={`/sample/${selected.folder}/${selected.sample}`}
                autoPlay loop muted playsInline
                style={{ width: '100%', aspectRatio: '4/3', objectFit: 'cover', display: 'block' }}
              />

              <div style={{
                padding: '8px 16px',
                background: `linear-gradient(to right, ${c.bg}, transparent)`,
                borderTop: `1px solid ${c.border}`,
                fontSize: 11, color: c.text, opacity: 0.8
              }}>
                ↑ Study this sign, then perform it in front of your webcam →
              </div>
            </motion.div>
          )
        })()}
      </AnimatePresence>

      {!selected && (
        <div style={{
          borderRadius: 16, border: '1px dashed rgba(99,102,241,0.15)',
          padding: '32px 16px', textAlign: 'center',
          background: 'rgba(99,102,241,0.02)'
        }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>🤟</div>
          <p style={{ fontSize: 12, color: '#334155' }}>Select a sign above to see the reference video</p>
        </div>
      )}
    </div>
  )
}
