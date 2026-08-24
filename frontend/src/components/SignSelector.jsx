import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { PlayCircle, ChevronRight } from 'lucide-react'

export default function SignSelector({ onSelect, selected }) {
  const [signs, setSigns] = useState([])
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    fetch('/api/signs').then(r => r.json()).then(setSigns).catch(() => setSigns([]))
  }, [])

  const handleSelect = (sign) => {
    onSelect(sign)
    setPlaying(false)
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Sign grid */}
      <div className="rounded-2xl border border-white/5 bg-[#0d0d14] p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[13px] font-semibold text-white">Choose a Sign</h2>
          <span className="text-[11px] text-zinc-500">{signs.length} signs</span>
        </div>
        <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto pr-1 scrollbar-thin">
          {signs.map(sign => (
            <motion.button
              key={sign.label}
              onClick={() => handleSelect(sign)}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-all duration-150 ${
                selected?.label === sign.label
                  ? 'bg-violet-600 border-violet-500 text-white shadow-lg shadow-violet-900/30'
                  : 'bg-white/3 border-white/5 text-zinc-400 hover:border-violet-500/50 hover:text-violet-300'
              }`}
            >
              {sign.label}
            </motion.button>
          ))}
          {signs.length === 0 && (
            <p className="text-zinc-600 text-[12px] p-2">No signs found — check verification_set/</p>
          )}
        </div>
      </div>

      {/* Reference video */}
      <AnimatePresence>
        {selected && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="rounded-2xl border border-white/5 bg-[#0d0d14] overflow-hidden"
          >
            <div className="px-4 pt-3 pb-2 flex items-center justify-between">
              <div>
                <div className="text-[11px] text-zinc-500 uppercase tracking-wider mb-0.5">Reference</div>
                <div className="text-[15px] font-semibold text-white flex items-center gap-1.5">
                  {selected.label}
                  <ChevronRight size={14} className="text-zinc-600" />
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-violet-400">
                <PlayCircle size={13} />
                Watch & imitate
              </div>
            </div>
            <video
              key={selected.label}
              src={`/sample/${selected.folder}/${selected.sample}`}
              autoPlay loop muted playsInline
              className="w-full aspect-video object-cover"
            />
            <div className="px-4 py-2.5 bg-violet-950/20 border-t border-violet-500/10">
              <p className="text-[11px] text-violet-300/70">
                ↑ Study this sign, then perform it in front of your webcam →
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!selected && (
        <div className="rounded-2xl border border-dashed border-white/5 p-6 text-center">
          <div className="text-3xl mb-2">🤟</div>
          <p className="text-[13px] text-zinc-600">Select a sign above to see the reference video</p>
        </div>
      )}
    </div>
  )
}
