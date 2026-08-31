/** Kannada TTS voices — UI ids shared across Sarvam, Parler, and MMS labels. */
export const VOICES = [
  { id: "female_clear", name: "Ananya",  role: "Clear female", tone: "#1d4ed8", hint: "Everyday speaking voice" },
  { id: "female_warm",  name: "Kavitha", role: "Warm female",  tone: "#7c3aed", hint: "Softer and slower" },
  { id: "male_clear",   name: "Suresh",  role: "Clear male",   tone: "#0f766e", hint: "Confident, moderate pace" },
  { id: "male_deep",    name: "Ramesh",  role: "Deep male",    tone: "#b45309", hint: "Lower and steady" },
  { id: "neutral",      name: "Neutral", role: "Neutral",      tone: "#475569", hint: "Plain studio voice" },
]

export const TTS_ENGINES = [
  { id: "auto",   name: "Auto",          hint: "Sarvam → Parler → MMS" },
  { id: "sarvam", name: "Sarvam Bulbul", hint: "Best Kannada · cloud" },
  { id: "parler", name: "Indic Parler",  hint: "Local · GPU" },
  { id: "mms",    name: "MMS fast",      hint: "Single voice · instant" },
]

export const DEFAULT_VOICE = "female_clear"
export const DEFAULT_ENGINE = "auto"
export const VOICE_SAMPLE_KN = "ನಮಸ್ಕಾರ"

export function voiceById(id) {
  return VOICES.find((v) => v.id === id) || VOICES[0]
}

export function engineById(id) {
  return TTS_ENGINES.find((e) => e.id === id) || TTS_ENGINES[0]
}
