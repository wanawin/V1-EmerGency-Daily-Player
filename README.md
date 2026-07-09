# Emergency Daily Player V1.5 — Decision Layer Audit

This version keeps all 6 emergency lanes. It keeps the old final playlist renamed as merged lane candidates and adds decision-layer strategy outputs.

New outputs:

- `07_DECISION_LAYER_STRATEGY_SUMMARY.csv`
- `07_DECISION_LAYER_STRATEGY_SUMMARY.txt`
- `07_DECISION_LAYER_STRATEGIES/*.csv`

Decision strategies compare:

- A: all lane Top30 rows capped at 50
- B: only streams overlapping in 2+ lanes
- C: overlap 2+ streams, keep all lane-qualified members
- D: overlap3 first, then overlap2, then single-lane best ranks
- E: one best row per overlap2+ stream
- F: merged candidates by best lane rank

Root-profile fallback remains in place: profile CSVs may be in `profiles/` or at repo root.
