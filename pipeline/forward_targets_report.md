# Forward Targets Report

Objective: build leakage-free Day+1, Day+2, Day+3 targets, aligned to feature_matrix.parquet's existing rows, to enable 1/2/3-day-ahead forecasting.

## Bug found and fixed in this stage

The first version of this script used `.unstack("machine").shift(-h).stack().reindex(...)` to
produce the forward-shifted columns. This silently produced an **all-NaN** result for every
single `Day_Plus_N__*` column -- confirmed by directly inspecting `forward_targets.parquet`'s
values (100% NaN), not just by re-reading the code and assuming it was correct. Downstream, this
would have caused every model-comparison run to be skipped for lack of a usable target, which is
exactly what happened on the first `run_modeling.py` attempt (9/9 "SKIPPED: train_rows=0").

**Fix**: the function now sorts `machine_day_targets.csv` by (machine, date) and applies a plain
`groupby("machine").shift(-h)` -- no unstack/stack/reindex gymnastics. A sanity assertion confirms
each machine's rows are one-per-calendar-day with no gaps (so "shift by h rows" is equivalent to
"shift by h days"), and a spot check below confirms a sampled row's `Day_Plus_1` value exactly
matches the next calendar day's actual same-day `A_any_eligible_breakdown` value.

## Methodology

1. Start from feature_matrix.parquet's existing 964-row frame (one row per
   historical prediction point). No future rows are introduced.
2. For each same-day target column (`A_any_eligible_breakdown`, `B_<family>_occurred`,
   `C_<family>_event_count`), create a Day+N version via `groupby(machine).shift(-N)` on the
   date-sorted target table, then re-key onto the feature matrix's own rows.
3. The last N rows of each machine's history have no day N days later within the dataset, so they
   are explicit NaNs for a Day+N target, never a coerced value.
4. Electrical/Mechanical Day+N targets are derived from the authoritative Level1->Level2 family
   mapping (hard-coded from label_taxonomy.py / the target definition report):

   Electrical families: ['Fan & Separator Mechanical/Electrical Failures', 'Electrical Protection, Instrumentation & Control Faults', 'Bag House / Bag Filter / Dust Collection', 'Motor & Electrical Drive Failures', 'Conveyor & Feeder Mechanical Failures', 'Hydraulic & Pneumatic System Failures', 'Bearing, Gearbox & Lubrication Failures', 'Other Confirmed Electrical Failures', 'Mill / Roller / Structural Mechanical Failures']
   Mechanical families: ['Bearing, Gearbox & Lubrication Failures', 'Conveyor & Feeder Mechanical Failures', 'Bag House / Bag Filter / Dust Collection', 'Hydraulic & Pneumatic System Failures', 'Other Confirmed Mechanical Failures', 'Mill / Roller / Structural Mechanical Failures', 'Electrical Protection, Instrumentation & Control Faults', 'Motor & Electrical Drive Failures']

   The Day+N Electrical target is 1 if ANY of that day's electrical-family `B_*_occurred` columns
   is 1 (else 0). Likewise for Mechanical.

## Verification performed

- Row count is unchanged after the merge (964 rows in, 964 rows out).
- `Day_Plus_1__A_any_eligible_breakdown` has 962 non-NaN values (not all-NaN as in the buggy version).
- Spot check: a sampled row's `Day_Plus_1__A_any_eligible_breakdown` value was confirmed to exactly
  equal the next calendar day's actual same-day `A_any_eligible_breakdown` value, read directly from
  `machine_day_targets.csv`.

## Forward Target Matrix

964 rows x 79 columns, aligned to feature_matrix.parquet's existing rows.

| Horizon | Non-NaN rows | Positive rate (non-NaN) |
|---|---|---|
| Day+1 | 962 | 28.90% |
| Day+2 | 960 | 28.96% |
| Day+3 | 958 | 29.02% |

## Columns produced

- `Day_Plus_N__A_any_eligible_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__A_electrical_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__A_mechanical_breakdown`: binary, N in {1, 2, 3}
- `Day_Plus_N__B_<family>_occurred`: binary per level-2 family, N in {1, 2, 3}
- `Day_Plus_N__C_<family>_event_count`: int per level-2 family, N in {1, 2, 3}
