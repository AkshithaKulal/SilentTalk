import { useState, useEffect } from "react"
import { X, Play } from "lucide-react"

const CATEGORIES = {
  greeting: { labels: ["hello", "thank you", "please", "sorry", "good morning", "how are you", "bye", "good afternoon", "good evening", "good night", "alright", "pleased"], name: "Greetings" },
  pronoun: { labels: ["i", "you", "he", "she", "we", "they", "it", "you (plural)"], name: "Pronouns" },
  people: { labels: ["mother", "father", "brother", "sister", "friend", "husband", "wife", "son", "daughter", "grandfather", "grandmother", "parent", "neighbour", "baby", "boy", "girl", "man", "woman", "child", "adult"], name: "People" },
  place: { labels: ["house", "school", "hospital", "park", "restaurant", "market", "office", "university", "library", "temple", "bank", "court", "city", "india", "ground", "store or shop", "street or road", "train station"], name: "Places" },
  time: { labels: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night", "week", "month", "year", "hour", "minute", "second", "time"], name: "Time" },
}

const getCat = (label) => {
  const l = label.toLowerCase()
  for (const [key, cat] of Object.entries(CATEGORIES)) {
    if (cat.labels.includes(l)) return { key, ...cat }
  }
  return { key: "all", name: "Other" }
}

export default function SignSelector({ onSelect, selected, onClose }) {
  const [signs, setSigns] = useState([])
  const [filter, setFilter] = useState("all")

  useEffect(() => {
    fetch("/api/signs").then((r) => r.json()).then(setSigns).catch(() => setSigns([]))
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  const filtered = filter === "all" ? signs : signs.filter((s) => getCat(s.label).key === filter)

  return (
    <div
      role="dialog"
      aria-label="Practice signs"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        background: "rgba(17,24,39,0.28)",
        display: "flex",
        justifyContent: "flex-end",
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(420px, 100%)",
          height: "100%",
          background: "var(--surface)",
          borderLeft: "1px solid var(--line)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div style={{ padding: "16px 18px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>Practice signs</div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>
              Watch a reference, then sign it on camera
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              marginLeft: "auto",
              width: 32,
              height: 32,
              border: "1px solid var(--line)",
              borderRadius: 8,
              background: "var(--surface)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ padding: "12px 18px", display: "flex", gap: 6, flexWrap: "wrap", borderBottom: "1px solid var(--line)" }}>
          {["all", ...Object.keys(CATEGORIES)].map((key) => {
            const label = key === "all" ? "All" : CATEGORIES[key].name
            const on = filter === key
            return (
              <button
                key={key}
                type="button"
                onClick={() => setFilter(key)}
                style={{
                  padding: "5px 10px",
                  borderRadius: 999,
                  border: `1px solid ${on ? "var(--accent)" : "var(--line)"}`,
                  background: on ? "var(--accent-soft)" : "transparent",
                  color: on ? "var(--accent-ink)" : "var(--muted)",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {label}
              </button>
            )
          })}
        </div>

        <div style={{ padding: 16, overflowY: "auto", flex: 1 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
            {filtered.map((sign) => {
              const active = selected?.label === sign.label
              return (
                <button
                  key={sign.label}
                  type="button"
                  onClick={() => onSelect(sign)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: 8,
                    border: `1px solid ${active ? "var(--accent)" : "var(--line)"}`,
                    background: active ? "var(--accent)" : "var(--surface)",
                    color: active ? "#fff" : "var(--ink)",
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {sign.label}
                </button>
              )
            })}
          </div>

          {selected ? (
            <div style={{ border: "1px solid var(--line)", borderRadius: 12, overflow: "hidden" }}>
              <div style={{ padding: "10px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: 16, fontWeight: 700 }}>{selected.label}</div>
                <span style={{ fontSize: 12, color: "var(--muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Play size={11} /> Reference
                </span>
              </div>
              <video
                key={selected.label}
                src={`/sample/${selected.folder}/${selected.sample}`}
                autoPlay
                loop
                muted
                playsInline
                style={{ width: "100%", aspectRatio: "4/3", objectFit: "cover", display: "block", background: "#111" }}
              />
            </div>
          ) : (
            <p style={{ fontSize: 13, color: "var(--muted)" }}>Select a word to watch how it is signed.</p>
          )}
        </div>
      </div>
    </div>
  )
}
