import json, re
from pathlib import Path
from collections import Counter

landmarks = Path("isl_recognition/landmarks")
label_counts = Counter()

for jf in landmarks.glob("*.json"):
    try:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        label = re.sub(r"^\d+\.\s*", "", meta.get("label", "")).strip()
        if label:
            label_counts[label] += 1
    except Exception:
        pass

problem = ["I", "Brother", "Tomorrow", "Yesterday", "Sister",
           "Night", "Saturday", "Grandmother", "Today", "Monday", "Friday",
           "Evening", "Afternoon", "Morning"]

print("=== PROBLEM SIGNS & CONFUSERS ===")
print(f"{'Sign':<30} {'Samples':>8}")
print("-" * 40)
for sign in sorted(label_counts.keys()):
    if any(p.lower() == sign.lower() for p in problem):
        print(f"{sign:<30} {label_counts[sign]:>8}")

print()
print(f"Total classes : {len(label_counts)}")
print(f"Total samples : {sum(label_counts.values())}")
print()
print("=== ALL CLASSES (sorted by count, ascending) ===")
for label, count in sorted(label_counts.items(), key=lambda x: x[1]):
    bar = "#" * (count // 2)
    print(f"  {label:<32} {count:>4}  {bar}")
