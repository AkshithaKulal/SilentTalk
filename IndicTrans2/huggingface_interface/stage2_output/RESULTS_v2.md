# Stage 2 Results v2

## Run identity
- Mode: Synthetic degraded fallback run
- Plain-language mode summary: Synthetic degraded fallback mode: English->Tulu continuity data (not a real KN-TCY run).
- Resume source: stage1_output/checkpoint-1500
- Not resumed from discarded stage2_output/checkpoint-3000 (different non-standard tag setup)
- Adapter evaluated: stage2_synthetic_sat_output
- Direction: eng_Latn -> sat_Olck (Tulu via sat_Olck alias)
- Evaluated samples: 100

## Metrics
- BLEU: 0.09
- chrF: 7.51

## Conclusion
- Corrected-tag run confirms checkpoint-3000's finding - the limitation is data quality (synthetic-only), not the tag implementation bug. Real KN-TCY data remains the required next step, not further training on existing data.

## Status
- Stage 2 work is paused here pending real KN-TCY data arrival.
