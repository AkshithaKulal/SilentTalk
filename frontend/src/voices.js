/** Kannada TTS voices — matches app.py VOICE_PRESETS (indic-parler-tts). */
export const VOICES = [
  { id: "female_clear", name: "Ananya",  role: "Clear female", tone: "#1d4ed8", hint: "Everyday speaking voice" },
  { id: "female_warm",  name: "Kavitha", role: "Warm female",  tone: "#7c3aed", hint: "Softer and slower" },
  { id: "male_clear",   name: "Suresh",  role: "Clear male",   tone: "#0f766e", hint: "Confident, moderate pace" },
  { id: "male_deep",    name: "Ramesh",  role: "Deep male",    tone: "#b45309", hint: "Lower and steady" },
  { id: "neutral",      name: "Neutral", role: "Neutral",      tone: "#475569", hint: "Plain studio voice" },
]

export const DEFAULT_VOICE = "female_clear"
export const VOICE_SAMPLE_KN = "ನಮಸ್ಕಾರ"

export function voiceById(id) {
  return VOICES.find((v) => v.id === id) || VOICES[0]
}
