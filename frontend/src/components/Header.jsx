import { motion } from 'framer-motion'
import { Hand } from 'lucide-react'

const StatusPill = ({ ok, label }) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '4px 10px', borderRadius: 20,
    background: ok ? 'rgba(5,150,105,0.08)' : '#f1f5f9',
    border: `1px solid ${ok ? 'rgba(5,150,105,0.2)' : '#e2e8f0'}`,
  }}>
    <span style={{
      width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
      background: ok ? '#059669' : '#cbd5e1',
      boxShadow: ok ? '0 0 6px rgba(5,150,105,0.5)' : 'none',
    }} />
    <span style={{ fontSize: 11, fontWeight: 600, color: ok ? '#065f46' : '#94a3b8' }}>{label}</span>
  </div>
)

export default function Header({ status }) {
  const allOk = status.classifier && status.translation_model && status.lora_adapter && status.tts_model
  return (
    <header style={{
      background: '#ffffff',
      borderBottom: '1px solid #e2e8f0',
      boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
      position: 'sticky', top: 0, zIndex: 50,
    }}>
      <div style={{
        maxWidth: 1600, margin: '0 auto',
        padding: '0 20px', height: 60,
        display: 'flex', alignItems: 'center', gap: 16,
      }}>
        {/* Logo */}
        <motion.div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}
          initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.35 }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10,
            background: 'linear-gradient(135deg, #0d9488, #059669)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(13,148,136,0.3)',
            flexShrink: 0,
          }}>
            <Hand size={18} color="white" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ fontSize: 17, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.4px', lineHeight: 1.1 }}>
              SilentTalk
            </div>
            <div style={{ fontSize: 10, color: '#64748b', fontWeight: 500 }}>ISL → Kannada</div>
          </div>
        </motion.div>

        {/* Divider */}
        <div style={{ width: 1, height: 24, background: '#e2e8f0', flexShrink: 0 }} />

        {/* Status pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <StatusPill ok={status.classifier}  label="Classifier" />
          <StatusPill ok={status.translation_model && status.lora_adapter} label="Translation" />
          <StatusPill ok={status.tts_model}   label="TTS" />
        </div>

        {/* All systems badge */}
        <div style={{ marginLeft: 'auto', flexShrink: 0 }}>
          <motion.div
            animate={{ opacity: allOk ? [1, 0.7, 1] : 1 }}
            transition={{ duration: 2.5, repeat: allOk ? Infinity : 0 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 14px', borderRadius: 20,
              background: allOk ? 'linear-gradient(135deg, #0d9488, #059669)' : '#f1f5f9',
              border: `1px solid ${allOk ? 'transparent' : '#e2e8f0'}`,
              boxShadow: allOk ? '0 2px 10px rgba(13,148,136,0.25)' : 'none',
            }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
              background: allOk ? 'rgba(255,255,255,0.8)' : '#cbd5e1',
            }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: allOk ? 'white' : '#94a3b8' }}>
              {allOk ? 'All systems ready' : 'Initializing...'}
            </span>
          </motion.div>
        </div>
      </div>
    </header>
  )
}
