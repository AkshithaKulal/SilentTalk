import { motion, AnimatePresence } from 'framer-motion'
import { Volume2, Loader2, Clock } from 'lucide-react'

const ConfBar = ({ pct }) => {
  const color = pct >= 70 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'
  return (
    <div className="h-1.5 rounded-full bg-white/5 overflow-hidden w-24">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        style={{ backgroundColor: color }}
        className="h-full rounded-full"
      />
    </div>
  )
}

export default function PredictionPanel({ prediction, translation, history, onSpeak, isSpeaking }) {
  const conf = prediction?.top_conf ?? 0
  const confColor = conf >= 70 ? 'text-green-400' : conf >= 40 ? 'text-amber-400' : 'text-red-400'
  const confLabel = conf >= 70 ? 'High confidence' : conf >= 40 ? 'Medium confidence' : 'Low confidence'

  return (
    <div className="flex flex-col gap-4">
      {/* Main prediction */}
      <div className="rounded-2xl border border-white/5 bg-[#0d0d14] p-5">
        <h2 className="text-[13px] font-semibold text-white mb-4">Prediction</h2>
        <AnimatePresence mode="wait">
          {!prediction ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center gap-3 py-8"
            >
              <div className="w-16 h-16 rounded-2xl border-2 border-dashed border-white/10 flex items-center justify-center text-2xl">
                🤟
              </div>
              <p className="text-[13px] text-zinc-600 text-center">Perform a sign and capture<br />to see prediction here</p>
            </motion.div>
          ) : (
            <motion.div
              key={prediction.top_label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-4"
            >
              {/* Big result */}
              <div className="text-center py-4 rounded-xl bg-white/2 border border-white/5">
                <motion.div
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  className="text-4xl font-black text-white mb-1"
                >
                  {prediction.top_label}
                </motion.div>
                <div className={`text-[13px] font-semibold ${confColor}`}>
                  {prediction.top_conf}% — {confLabel}
                </div>
              </div>

              {/* Top-5 bars */}
              {prediction.top5 && (
                <div className="space-y-2">
                  <p className="text-[11px] text-zinc-500 uppercase tracking-wider">Top 5</p>
                  {prediction.top5.map((p, i) => (
                    <div key={p.label} className="flex items-center gap-2.5">
                      <span className="text-[11px] text-zinc-600 w-3">{i + 1}</span>
                      <span className="text-[12px] text-zinc-300 flex-1 truncate">{p.label}</span>
                      <ConfBar pct={p.conf} />
                      <span className="text-[11px] text-zinc-500 w-10 text-right">{p.conf}%</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Translation */}
      <AnimatePresence>
        {translation && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-violet-500/20 bg-violet-950/20 p-5"
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-[11px] text-violet-400/60 uppercase tracking-wider mb-0.5">Kannada Translation</p>
                <p className="text-[10px] text-zinc-600">checkpoint-1500 fine-tuned</p>
              </div>
              <motion.button
                onClick={onSpeak}
                disabled={isSpeaking}
                whileHover={!isSpeaking ? { scale: 1.05 } : {}}
                whileTap={!isSpeaking ? { scale: 0.95 } : {}}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600/20 border border-violet-500/30 text-violet-300 text-[12px] font-medium hover:bg-violet-600/30 disabled:opacity-50 transition-all"
              >
                {isSpeaking ? <Loader2 size={12} className="animate-spin" /> : <Volume2 size={12} />}
                {isSpeaking ? 'Speaking...' : 'Speak'}
              </motion.button>
            </div>
            <p className="kannada text-2xl font-bold text-violet-200 leading-relaxed">
              {translation}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* History */}
      <div className="rounded-2xl border border-white/5 bg-[#0d0d14] p-4">
        <div className="flex items-center gap-2 mb-3">
          <Clock size={13} className="text-zinc-600" />
          <h3 className="text-[13px] font-semibold text-white">Session History</h3>
          {history.length > 0 && (
            <span className="ml-auto text-[11px] text-zinc-600">{history.length} predictions</span>
          )}
        </div>
        {history.length === 0 ? (
          <p className="text-[12px] text-zinc-700 text-center py-3">No predictions yet</p>
        ) : (
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {history.map((h, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-start justify-between gap-2 rounded-lg bg-white/2 px-3 py-2"
              >
                <div className="min-w-0">
                  <span className="text-[12px] font-semibold text-zinc-200">{h.label}</span>
                  {h.translation && (
                    <span className="kannada text-[11px] text-violet-400 ml-2">{h.translation}</span>
                  )}
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-[11px] text-zinc-500">{h.conf}%</div>
                  <div className="text-[10px] text-zinc-700">{h.time}</div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
