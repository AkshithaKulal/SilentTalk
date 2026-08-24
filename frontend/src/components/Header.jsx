import { motion } from 'framer-motion'
import { HandMetal } from 'lucide-react'

const StatusDot = ({ ok, label }) => (
  <span className="flex items-center gap-1.5 text-[11px]">
    <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : 'bg-slate-600'}`} />
    <span className={ok ? 'text-slate-300' : 'text-slate-600'}>{label}</span>
  </span>
)

export default function Header({ status }) {
  const allOk = status.classifier && status.translation_model && status.lora_adapter && status.tts_model
  return (
    <header style={{
      background: 'linear-gradient(to right, #060912, #0d1117, #060912)',
      borderBottom: '1px solid rgba(99,102,241,0.15)'
    }}>
      <div className="max-w-[1600px] mx-auto px-5 py-3 flex items-center gap-5 flex-wrap">

        {/* Logo */}
        <motion.div className="flex items-center gap-3"
          initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 16px rgba(99,102,241,0.4)'
          }}>
            <HandMetal size={18} color="white" />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, letterSpacing: '-0.3px',
              background: 'linear-gradient(135deg, #a5b4fc, #818cf8)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              SilentTalk
            </div>
            <div style={{ fontSize: 10, color: '#475569', lineHeight: 1 }}>Indian Sign Language → Kannada</div>
          </div>
        </motion.div>

        <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.07)' }} />

        {/* Status */}
        <div className="flex items-center gap-4">
          <StatusDot ok={status.classifier} label="Classifier" />
          <StatusDot ok={status.translation_model && status.lora_adapter} label="Translation" />
          <StatusDot ok={status.tts_model} label="TTS" />
        </div>

        <div className="ml-auto">
          <motion.div
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
            style={{
              fontSize: 11, padding: '4px 12px', borderRadius: 20,
              border: `1px solid ${allOk ? 'rgba(52,211,153,0.3)' : 'rgba(99,102,241,0.2)'}`,
              background: allOk ? 'rgba(52,211,153,0.06)' : 'rgba(99,102,241,0.06)',
              color: allOk ? '#6ee7b7' : '#818cf8',
            }}>
            {allOk ? '● All systems ready' : '◌ Initializing...'}
          </motion.div>
        </div>
      </div>
    </header>
  )
}
