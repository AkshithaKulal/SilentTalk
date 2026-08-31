import { Loader2, RotateCcw, Volume2 } from "lucide-react"

const confColor = (c) => (c >= 70 ? "var(--ok)" : c >= 40 ? "var(--warn)" : "var(--bad)")
const confBg = (c) => (c >= 70 ? "var(--ok-soft)" : c >= 40 ? "var(--warn-soft)" : "var(--bad-soft)")

export default function LiveRail({
  prediction,
  translation,
  translating,
  history,
  isSpeaking,
  speakingTarget,
  voices,
  selectedVoice,
  onVoiceChange,
  onSpeakWord,
  onReplayHistory,
}) {
  const conf = prediction?.top_conf ?? 0

  return (
    <aside className="live-rail" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <section
        style={{
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: 16,
          padding: 16,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--faint)",
            marginBottom: 12,
          }}
        >
          Listening
        </div>

        {!prediction ? (
          <p style={{ fontSize: 15, color: "var(--muted)", lineHeight: 1.5 }}>
            Start the camera and press Go Live. The current sign shows here; it joins the message when confidence is high.
          </p>
        ) : (
          <div>
            <div
              style={{
                padding: "14px 14px 12px",
                borderRadius: 12,
                background: confBg(conf),
                marginBottom: 12,
              }}
            >
              <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
                {prediction.top_label}
              </div>
              <div style={{ marginTop: 6, fontSize: 13, fontWeight: 600, color: confColor(conf) }}>
                {conf}% confidence
              </div>
              <div
                style={{
                  marginTop: 10,
                  height: 4,
                  borderRadius: 4,
                  background: "rgba(0,0,0,0.08)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${Math.min(conf, 100)}%`,
                    height: "100%",
                    background: confColor(conf),
                  }}
                />
              </div>
            </div>

            {(translating || translation) && (
              <p className="kannada" style={{ fontSize: 20, fontWeight: 700, marginBottom: 12, minHeight: 28 }}>
                {translating ? "…" : translation}
              </p>
            )}

            <button
              type="button"
              onClick={onSpeakWord}
              disabled={!prediction || isSpeaking}
              style={{
                width: "100%",
                height: 40,
                borderRadius: 10,
                border: "1px solid var(--line)",
                background: "var(--surface)",
                color: "var(--ink)",
                fontSize: 13,
                fontWeight: 600,
                cursor: !prediction || isSpeaking ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
                opacity: !prediction || isSpeaking ? 0.5 : 1,
              }}
            >
              {speakingTarget === "word" && isSpeaking ? (
                <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
              ) : (
                <Volume2 size={14} />
              )}
              Preview word
            </button>

            {prediction.top5 && (
              <ol style={{ listStyle: "none", marginTop: 14, display: "flex", flexDirection: "column", gap: 6 }}>
                {prediction.top5.map((p, i) => (
                  <li key={p.label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                    <span style={{ width: 14, color: "var(--faint)" }}>{i + 1}</span>
                    <span style={{ flex: 1, fontWeight: i === 0 ? 700 : 500, color: i === 0 ? "var(--ink)" : "var(--muted)" }}>
                      {p.label}
                    </span>
                    <span style={{ color: "var(--muted)", width: 36, textAlign: "right" }}>{p.conf}%</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </section>

      {voices?.length > 0 && (
        <section
          style={{
            background: "var(--surface)",
            border: "1px solid var(--line)",
            borderRadius: 16,
            padding: 14,
          }}
        >
          <div
            style={{
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              color: "var(--faint)",
              marginBottom: 10,
            }}
          >
            Voice
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {voices.map((v) => {
              const on = selectedVoice === v.id
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => onVoiceChange(v.id)}
                  title={v.description}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 999,
                    border: `1px solid ${on ? "var(--accent)" : "var(--line)"}`,
                    background: on ? "var(--accent-soft)" : "var(--surface)",
                    color: on ? "var(--accent-ink)" : "var(--muted)",
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {v.name}
                </button>
              )
            })}
          </div>
        </section>
      )}

      <section
        style={{
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: 16,
          padding: 14,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--faint)",
            marginBottom: 10,
          }}
        >
          Spoken
        </div>
        {history.length === 0 ? (
          <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.5 }}>
            After you speak, messages stay here so a listener can hear them again.
          </p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 240, overflowY: "auto" }}>
            {history.map((h, i) => (
              <div
                key={`${h.time}-${i}`}
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  background: "var(--bg)",
                  display: "flex",
                  gap: 8,
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{h.sentence}</div>
                  <div className="kannada" style={{ fontSize: 14, color: "var(--accent-ink)", marginTop: 2 }}>
                    {h.kannada}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 4 }}>{h.time}</div>
                </div>
                <button
                  type="button"
                  onClick={() => onReplayHistory(h.kannada)}
                  disabled={isSpeaking}
                  aria-label="Replay"
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    border: "1px solid var(--line)",
                    background: "var(--surface)",
                    cursor: isSpeaking ? "not-allowed" : "pointer",
                    color: "var(--accent)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  {speakingTarget === "replay" && isSpeaking && i === 0 ? (
                    <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                  ) : (
                    <RotateCcw size={14} />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </aside>
  )
}
