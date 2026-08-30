import requests, time

words = ["Friend", "Hello", "Mother", "Today", "Thank you", "House", "School", "Brother"]
BASE = "http://localhost:5000"

print("Translation benchmark (warm model):")
print("-" * 45)

# Warm up first
requests.post(f"{BASE}/api/translate", json={"text": "Hello"}, timeout=60)

for word in words:
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/api/translate", json={"text": word}, timeout=60)
    elapsed = (time.perf_counter() - t0) * 1000
    result = r.json().get("translation", "ERROR")
    print(f"  {word:<15} -> {result:<20} {elapsed:.0f}ms")
