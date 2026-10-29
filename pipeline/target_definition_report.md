# Prediction Target Definition Report

## Step 4.1 — Eligible Breakdown Classification

Authoritative rule: a Breakdown-sheet record is eligible for the primary electrical/mechanical failure target **only if** `Category` is exactly 'Electrical Breakdown' (code E01) or 'Mechanical Breakdown' (code M01). This is not a keyword guess — every record's Breakdown Name and Reason field carries this code as a literal text prefix, and per-prefix counts match the Category field's value counts exactly, confirming Category is the plant's own documented classification.

### Category → Code verification

| Category | Code | Event count |
|---|---|---|
| Utility (Electricity) | U01 | 590 |
| Planned Down Time | P01 | 418 |
| Others down time | O01 | 388 |
| Warehouse block | W01 | 230 |
| Electrical Breakdown | E01 | 230 |
| Mechanical Breakdown | M01 | 214 |
| In Process Setup/ Adjustment | I01 | 98 |
| Raw Material Shortage | R01 | 16 |

Eligible (E01+M01) unique name/reason combinations: 241 of 407 total. 15 are flagged 'Needs Review' (fell into a residual 'Other Confirmed ___' bucket with no matching family keyword — not force-classified).


## Step 4.2 — Failure-Family Taxonomy (3 levels)

Level 1: Electrical / Mechanical (from Category). Level 2: failure family (below, data-driven). Level 3: normalized breakdown reason (see event_level_reason_table.csv).

| Level 1 | Level 2 Family | Unique Reasons | Events | Low-confidence rows |
|---|---|---|---|---|
| Electrical | Fan & Separator Mechanical/Electrical Failures | 14 | 68 | 0 |
| Electrical | Electrical Protection, Instrumentation & Control Faults | 13 | 53 | 0 |
| Electrical | Bag House / Bag Filter / Dust Collection | 13 | 31 | 0 |
| Electrical | Motor & Electrical Drive Failures | 16 | 31 | 0 |
| Electrical | Conveyor & Feeder Mechanical Failures | 14 | 20 | 0 |
| Electrical | Hydraulic & Pneumatic System Failures | 4 | 9 | 0 |
| Electrical | Bearing, Gearbox & Lubrication Failures | 6 | 8 | 0 |
| Electrical | Other Confirmed Electrical Failures | 6 | 8 | 7 |
| Electrical | Mill / Roller / Structural Mechanical Failures | 1 | 2 | 0 |
| Mechanical | Bearing, Gearbox & Lubrication Failures | 15 | 60 | 0 |
| Mechanical | Conveyor & Feeder Mechanical Failures | 19 | 59 | 0 |
| Mechanical | Bag House / Bag Filter / Dust Collection | 15 | 52 | 0 |
| Mechanical | Hydraulic & Pneumatic System Failures | 11 | 24 | 0 |
| Mechanical | Other Confirmed Mechanical Failures | 8 | 9 | 8 |
| Mechanical | Mill / Roller / Structural Mechanical Failures | 3 | 5 | 0 |
| Mechanical | Electrical Protection, Instrumentation & Control Faults | 2 | 3 | 0 |
| Mechanical | Motor & Electrical Drive Failures | 2 | 2 | 0 |


## Event Granularity

Each row in the Breakdown sheet represents an INDIVIDUAL breakdown EVENT, not a shift-level aggregate. Evidence: up to 5 distinct rows share the same (Mill Name, Date, Shift) key, each with a different Category/Breakdown Name/Reason and its own Downtime value (range 5-480 minutes, consistent with within-shift event durations rather than full-shift totals). Therefore, 'number of breakdown events' for target C must be computed as ROW COUNTS per (machine, family, day) -- summing rows, not counting distinct shifts -- and the 18 rows sharing an identical (Mill, Date, Shift, Name, Reason) key found in the prior data-quality audit should be treated as likely duplicate entries (same event logged twice) and deduplicated before counting, not as 18 additional independent events.

Duplicate (Mill, Date, Shift, Name, Reason) rows removed before target construction: 18


## Step 4.3 — Target Definitions

Machine-day target table: 964 rows (964 rows x 26 columns), covering every (machine, calendar date) combination in range.

- **Target A** (`A_any_eligible_breakdown`): binary, any E/M breakdown that machine-day. Overall positive rate: 28.84%.

- **Target B** (`B_<family>_occurred`): binary per failure family, one column per family discovered in 4.2.

- **Target C** (`C_<family>_event_count`): integer count of events per family per machine-day (row counts, post-deduplication, per the granularity finding).

- **Target D**: event-level reason table (`event_level_reason_table.csv`) — most-likely detailed reason, modeled conditional on the predicted family, not as a flat 200+-way classification.

- **Target E** (`E_downtime_minutes_total/max`): see feasibility check below — only modeled as a regression where the data supports it.


## Target E Feasibility (downtime)

7 of 10 failure families have >=30 events AND a coefficient of variation < 1.2, the minimum bar for a potentially reliable downtime regression. Families below this bar should NOT have a downtime model forced onto them -- downtime should be reported only as a historical descriptive range for those families, not a point prediction.

| Family | Events | Mean (min) | Std | CV | Reliable for regression? |
|---|---|---|---|---|---|
| Bag House / Bag Filter / Dust Collection | 81 | 66.0 | 57.0 | 0.86 | True |
| Conveyor & Feeder Mechanical Failures | 79 | 126.3 | 93.1 | 0.74 | True |
| Bearing, Gearbox & Lubrication Failures | 67 | 142.6 | 151.2 | 1.06 | True |
| Fan & Separator Mechanical/Electrical Failures | 67 | 92.1 | 69.9 | 0.76 | True |
| Electrical Protection, Instrumentation & Control Faults | 56 | 119.6 | 132.6 | 1.11 | True |
| Hydraulic & Pneumatic System Failures | 33 | 122.0 | 121.3 | 0.99 | True |
| Motor & Electrical Drive Failures | 33 | 151.5 | 139.8 | 0.92 | True |
| Other Confirmed Mechanical Failures | 9 | 95.6 | 65.5 | 0.68 | False |
| Other Confirmed Electrical Failures | 8 | 91.2 | 114.7 | 1.26 | False |
| Mill / Roller / Structural Mechanical Failures | 6 | 93.3 | 78.5 | 0.84 | False |