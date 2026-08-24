import { useState, useEffect } from 'react'

export function useSystemStatus() {
  const [status, setStatus] = useState({ classifier: false, translation_model: false, lora_adapter: false, tts_model: false })

  useEffect(() => {
    const check = () => fetch('/api/status').then(r => r.json()).then(setStatus).catch(() => {})
    check()
    const id = setInterval(check, 15000)
    return () => clearInterval(id)
  }, [])

  return status
}
