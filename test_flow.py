#!/usr/bin/env python3
"""End-to-end flow test for SilentTalk."""
import requests, time, sys

BASE = "http://localhost:5000"

print("=" * 60)
print("SilentTalk End-to-End Flow Test")
print("=" * 60)

# 1. Status
print("\n1. /api/status")
try:
    r = requests.get(f"{BASE}/api/status", timeout=5)
    s = r.json()
    for k, v in s.items():
        status = "OK" if v else "NOT READY"
        print(f"   {k:<25} {status}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# 2. Voices
print("\n2. /api/voices")
try:
    r = requests.get(f"{BASE}/api/voices", timeout=5)
    d = r.json()
    if "voices" in d:
        for v in d["voices"]:
            flag = " <-- selected" if v["id"] == d.get("default") else ""
            print(f"   {v['id']:<15} {v['name']:<10} {v['description']}{flag}")
    else:
        print(f"   Response: {d}")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Translate
print("\n3. /api/translate  (text='Child')")
try:
    t0 = time.time()
    r = requests.post(f"{BASE}/api/translate",
                      json={"text": "Child"}, timeout=60)
    elapsed = time.time() - t0
    d = r.json()
    if "translation" in d:
        print(f"   OK in {elapsed:.1f}s  ->  {d['translation']}")
    else:
        print(f"   ERROR: {d}")
except Exception as e:
    print(f"   ERROR (timeout or connection): {e}")

# 4. TTS
print("\n4. /api/tts  (text='ಮಗು.', voice='female_clear')")
print("   NOTE: First call downloads ~1.5GB model - may take several minutes...")
try:
    t0 = time.time()
    r = requests.post(f"{BASE}/api/tts",
                      json={"text": "ಮಗು.", "voice": "female_clear"},
                      timeout=300)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        if "audio_b64" in d:
            size_kb = len(d["audio_b64"]) * 3 // 4 // 1024
            print(f"   OK in {elapsed:.1f}s  ->  ~{size_kb} KB WAV audio")
        else:
            print(f"   ERROR in response: {d}")
    else:
        print(f"   HTTP {r.status_code}: {r.text[:400]}")
except requests.exceptions.Timeout:
    elapsed = time.time() - t0
    print(f"   TIMEOUT after {elapsed:.0f}s - model still downloading, try again later")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("Done")
