import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, BookOpen, Search } from 'lucide-react'

const CATEGORIES = {
  greeting:  { labels: ['hello','thank you','please','sorry','good morning','how are you','bye','good afternoon','good evening','good night','alright','pleased'],
               color: '#0d9488', bg: 'rgba(13,148,136,0.08)', border: 'rgba(13,148,136,0.2)', name: 'Greetings' },
  pronoun:   { labels: ['i','you','he','she','we','they','it','you (plural)'],
               color: '#7c3aed', bg: 'rgba(124,58,237,0.08)', border: 'rgba(124,58,237,0.2)', name: 'Pronouns' },
  people:    { labels: ['mother','father','brother','sister','friend','husband','wife','son','daughter','grandfather','grandmother','parent','neighbour','baby','boy','girl','man','woman','child','adult'],
               color: '#d97706', bg: 'rgba(217,119,6,0.08)', border: 'rgba(217,119,6,0.2)', name: 'People' },
  place:     { labels: ['house','school','hospital','park','restaurant','market','office','university','library','temple','bank','court','city','india','ground','store or shop','street or road','train station'],
               color: '#059669', bg: 'rgba(5,150,105,0.08)', border: 'rgba(5,150,105,0.2)', name: 'Places' },
  time:      { labels: ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','today','tomorrow','yesterday','morning','afternoon','evening','night','week','month','year','hour','minute','second','time'],
               color: '#db2777', bg: 'rgba(219,39,119,0.08)', border: 'rgba(219,39,119,0.2)', name: 'Time' },
}

const getCat = (label) => {
  const l = label.toLowerCase()
  for (const [key, cat] of Object.entries(CATEGORIES)) {
    if (cat.labels.includes(l)) return { key, ...cat }
  }
  return { key: 'default', color: '#0d9488', bg: 'rgba(13,148,136,0.06)', border: 'rgba(13,148,136,0.15)', name: 'Other' }
}

export default function SignSelector({ onSelect, selected }) {
  const [signs, setSigns]   = useState([])
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    fetch('/api/signs').then(r => r.json()).then(setSigns).catch(() => setSigns([]))
  }, [])

  const categories = [...new Set(signs.map(s => getCat(s.label).key))]
  const filtered = filter === 'all' ? signs : signs.filter(s => getCat(s.label).key === filter)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* Sign grid card */}
      <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 16,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)', overflow: 'hidden' }}>

        {/* Header */}
        <div style={{ padding: '14px 16px 12px', borderBottom: '1px solid #f1f5f9' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 28, height: 28, borderRadius: 8, background: 'rgba(13,148,136,0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <BookOpen size={14} color="#0d9488" />
              </div>
              <span style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>Sign Library</span>
            </div>
            <span style={{ fontSize: 11, fontWeight: 600, color: '#64748b',
              background: '#f1f5f9', padding: '3px 8px', borderRadius: 12,
              border: '1px solid #e2e8f0' }}>
              {signs.length} signs
            </span>
          </div>

          {/* Category filter pills */}
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            <button onClick={() => setFilter('all')}
              style={{ padding: '4px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: 'pointer',
                background: filter === 'all' ? '#0d9488' : 'transparent',
                color: filter === 'all' ? 'white' : '#64748b',
                border: `1px solid ${filter === 'all' ? '#0d9488' : '#e2e8f0'}`,
                transition: 'all 0.15s' }}>
              All
            </button>
            {Object.entries(CATEGORIES).map(([key, cat]) => {
              const has = signs.some(s => getCat(s.label).key === key)
              if (!has) return null
              return (
                <button key={key} onClick={() => setFilter(filter === key ? 'all' : key)}
                  style={{ padding: '4px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: 'pointer',
                    background: filter === key ? cat.color : 'transparent',
                    color: filter === key ? 'white' : '#64748b',
                    border: `1px solid ${filter === key ? cat.color : '#e2e8f0'}`,
                    transition: 'all 0.15s' }}>
                  {cat.name}
                </button>
              )
            })}
          </div>
        </div>

        {/* Grid */}
        <div style={{ padding: '10px 12px', display: 'flex', flexWrap: 'wrap', gap: 6,
          maxHeight: 200, overflowY: 'auto' }}>
          {filtered.map(sign => {
            const cat = getCat(sign.label)
            const isActive = selected?.label === sign.label
            return (
              <motion.button key={sign.label} onClick={() => onSelect(sign)}
                whileHover={{ scale: 1.03, y: -1 }} whileTap={{ scale: 0.96 }}
                style={{
                  padding: '5px 11px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                  cursor: 'pointer', whiteSpace: 'nowrap', transition: 'all 0.15s',
                  background: isActive ? cat.color : cat.bg,
                  color: isActive ? 'white' : cat.color,
                  border: `1px solid ${isActive ? cat.color : cat.border}`,
                  boxShadow: isActive ? `0 2px 8px ${cat.bg}` : 'none',
                }}>
                {sign.label}
              </motion.button>
            )
          })}
          {filtered.length === 0 && (
            <p style={{ color: '#94a3b8', fontSize: 12, padding: '8px 4px' }}>No signs in this category</p>
          )}
        </div>
      </div>

      {/* Reference video */}
      <AnimatePresence mode="wait">
        {selected ? (() => {
          const cat = getCat(selected.label)
          return (
            <motion.div key={selected.label}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
              style={{ background: '#fff', borderRadius: 16, overflow: 'hidden',
                border: `1px solid ${cat.border}`,
                boxShadow: `0 4px 16px ${cat.bg}` }}>

              {/* Video header */}
              <div style={{ padding: '12px 14px 10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: cat.bg, borderBottom: `1px solid ${cat.border}` }}>
                <div>
                  <div style={{ fontSize: 10, fontWeight: 600, color: cat.color,
                    textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 2 }}>
                    Reference Sign
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: '#0f172a' }}>{selected.label}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11,
                  fontWeight: 600, color: cat.color,
                  background: '#fff', padding: '5px 10px', borderRadius: 20,
                  border: `1px solid ${cat.border}` }}>
                  <Play size={10} fill="currentColor" />
                  Watch & imitate
                </div>
              </div>

              <video key={selected.label} src={`/sample/${selected.folder}/${selected.sample}`}
                autoPlay loop muted playsInline
                style={{ width: '100%', aspectRatio: '4/3', objectFit: 'cover', display: 'block' }} />

              <div style={{ padding: '8px 14px', background: cat.bg, borderTop: `1px solid ${cat.border}`,
                fontSize: 11, fontWeight: 500, color: cat.color }}>
                📖 Study this sign, then perform it in front of your webcam
              </div>
            </motion.div>
          )
        })() : (
          <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{ background: '#fff', borderRadius: 16, border: '1px dashed #e2e8f0',
              padding: '32px 16px', textAlign: 'center',
              boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
            <div style={{ fontSize: 36, marginBottom: 8 }}>🤟</div>
            <p style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Choose a sign above</p>
            <p style={{ fontSize: 12, color: '#94a3b8' }}>Reference video will appear here</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
