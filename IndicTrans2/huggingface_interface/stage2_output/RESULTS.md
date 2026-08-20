# Stage 2 Results — SilentTalk Tulu MT

Run baseline for this branch resumes from stage1_output/checkpoint-1500.
Do not resume from discarded stage2_output/checkpoint-3000 (different, non-standard tag setup).

## Language-tag alias (required)

IndicTrans2 has **no** `tcy_Knda` token. [`train_lora.py`](../train_lora.py) skips pairs where `src_lang == tgt_lang`, so Tulu **cannot** reuse `kan_Knda` on both sides.

| Role | FLORES tag | Meaning in this project |
|------|------------|-------------------------|
| Source (Kannada) | `kan_Knda` | Real Kannada |
| Target (Tulu) | `sat_Olck` | **Alias only** — Tulu text in Kannada script, **not** Santali |

`IndicProcessor` maps `sat_Olck` → ISO `or` by default. Stage 2 train/inference force `sat_Olck` → `kn` via [`tulu_lang_alias.py`](../tulu_lang_alias.py) so Kannada-script Tulu is tokenized correctly.

There is **no public GitHub mirror** of DravidianLangTech-2022 KN–TCY. Real data must come from organizers (e.g. Dr. Asha) as `kn_tcy_raw.csv`.

Published ballpark (approximate, not a project KPI): ~23 BLEU for a plain Transformer on ~10k KN–TCY; higher figures (~41) depend on ATG / linguistic features.

---

## A. Synthetic EN→Tulu (historical, incomplete)

Mode statement: This is a synthetic degraded fallback run, not a real KN-TCY run.

- **Path:** `stage2_output/checkpoint-3000`
- **Direction:** `eng_Latn` → `sat_Olck` with alias override to Kannada normalization
- **Data:** ~19.5k synthetic backtranslation pairs (`synthetic_en_tulu_fixed.csv`)
- **best_global_step:** 3000
- **best_metric (eval_BLEU):** **0.839**
- **Final eval_chrF (step 3000):** 14.47
- **Final eval_loss (step 3000):** 4.47

| Step | eval_BLEU | eval_chrF | eval_loss |
|------|-----------|-----------|-----------|
| 500 | 0.809 | 15.16 | 5.23 |
| 1000 | 0.676 | 13.73 | 4.82 |
| 1500 | 0.640 | 12.91 | 4.67 |
| 2000 | 0.816 | 14.44 | 4.59 |
| 2500 | 0.810 | 15.10 | 4.53 |
| 3000 | **0.839** | 14.47 | 4.47 |

**Do not treat this as the final Tulu model.** It is not real KN→Tulu. Further synthetic-only training is unlikely to reach competitive quality.

---

## B. Real KN→Tulu (current path)

Mode statement: This section is for real KN-TCY runs from organizer-provided parallel data.

- **Prepare:** `python prepare_stage2_data.py --mode real` (needs `kn_tcy_raw.csv`)
- **Train:** `.\run_stage2_train.ps1` → output `stage2_kn_tcy_output/`
- **Resume:** base `indictrans2-en-indic-1B` + Stage 1 `stage1_output/checkpoint-1500` (never discarded stage2_output/checkpoint-3000)
- **Tags:** `kan_Knda` → `sat_Olck` (Tulu alias)
- **Eval:** `python test_stage2_quality.py` / `python test_stage2_inference.py`

### Metrics

_Pending — run training after `kn_tcy_raw.csv` arrives, then fill BLEU/chrF here._

### Status

Blocked on real parallel data from organizers. Pipeline and tag alias are ready.

---

## C. Synthetic degraded fallback (optional continuity)

If `kn_tcy_raw.csv` is unavailable:

```powershell
python prepare_stage2_data.py --mode synthetic_degraded
.\run_stage2_train_synthetic.ps1
```

Writes `eng_Latn-sat_Olck` under `stage2_data/` and trains into `stage2_synthetic_sat_output/`. **Documented as non-competitive** — not a substitute for real KN–TCY.
