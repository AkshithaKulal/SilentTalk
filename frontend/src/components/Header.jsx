import { motion } from 'framer-motion'
import { Hand } from 'lucide-react'

const DOT = ({ ok }) => (
  <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${ok ? 'bg-green-400' : 'bg-zinc-600'}`} />
)

export default function Header({ status }) {
  const allOk = status.classifier && status.translation_model && status.lora_adapter && status.tts_model
  return (
    <header className="border-b border-white/5 bg-[#0d0d14] px-5 py-3">
      <div className="max-w-[1600px] mx-auto flex items-center gap-4 flex-wrap">
        {/* Logo */}
        <motion.div
          className="flex items-center gap-2.5"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="w-8 h-8 rounded-lg bg-violet-600/20 flex items-center justify-center border border-violet-500/30">
            <Hand size={16} className="text-violet-400" />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-tight">SilentTalk</div>
            <div className="text-[10px] text-zinc-500 leading-none">ISL → Kannada</div>
          </div>
        </motion.div>

        <div className="h-5 w-px bg-white/10" />

        {/* Status indicators */}
        <div className="flex items-center gap-4 text-[11px] text-zinc-400">
          <span><DOT ok={status.classifier} />Classifier</span>
          <span><DOT ok={status.translation_model && status.lora_adapter} />Translation</span>
          <span><DOT ok={status.tts_model} />TTS</span>
        </div>

        <div className="ml-auto">
          <div className={`text-[11px] px-2.5 py-1 rounded-full border ${allOk ? 'border-green-500/30 text-green-400 bg-green-500/5' : 'border-zinc-700 text-zinc-500 bg-white/2'}`}>
            {allOk ? '● All systems ready' : '○ Loading...'}
          </div>
        </div>
      </div>
    </header>
  )
}
