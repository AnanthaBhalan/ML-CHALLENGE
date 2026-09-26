# Amazon ML Challenge 2026 — Methodology Document

**Team Name:** AnanthaBhalan
**Team Members:** AnanthaBhalan

## Methodology Used

A two-stage Entity Resolution pipeline for ~23M records of highly skewed,
Zipfian-distributed, noisy business names:

1. **High-recall multi-key blocking** — a de-duplicated union of three
   deterministic keys (noise-stripped name prefix, exact cleaned name, and
   address-digit prefix), each country-scoped. Recovers **95.09%** of true pairs.
2. **Precision-weighted pair classification** — LightGBM on 11 RapidFuzz/Polars
   similarity features, with the decision threshold tuned directly against
   **macro F₀.₅** and negatives sampled to match inference density.

## Candidate Generation / Blocking Strategy

Initial EDA established that pure equality blocking cannot reach the target
volume, and — importantly — *why*. Business names follow a Zipfian
distribution: 52 tokens (`limited`, `private`, `llc`, `pvt`, `dr`, …) appear in
**92.2%** of all documents, and the top 500 name-prefix buckets hold 50.3% of
all S2/S3 records. Stripping noise tokens does not fix this; it *shifts* the
mass. Measured directly: removing leading noise tokens raised recall 77.1% →
83.2% but **increased** candidates 7,410 → 7,744/S1, because the records simply
redistributed into the next-commonest token (`pediatric`, `international`).
The concentration is a property of the data, not of the key.

We measured eight schemes against `train_ground_truth.tsv` and mapped the full
frontier. Every equality key lies on **one** curve; no point on it delivers
>90% recall at <30 candidates/S1. Best options:

| Scheme | Pair recall | Entity recall | F₀.₅ ceiling | Cand/S1 |
|---|---|---|---|---|
| `k0` name prefix (4 chars) | 77.1% | 47.9% | 0.8980 | 7,410 |
| `k1` exact cleaned name | 47.2% | — | — | 77 |
| postal ∩ name prefix | 59.9% | 23.9% | 0.7622 | 26 |
| **UNION `k0`+`k1`+`k2` (shipped)** | **95.09%** | **86.27%** | **0.9811** | **19,244** |

**On the three "levers" — what worked and what did not.** We implemented and
benchmarked each, and report them honestly rather than claiming all three
contributed to the shipped pool:

1. **Exact out-of-core IDF damping** — *worked, but confirmed prior work.*
   Computed exact document frequencies over the 10M S2/S3 records natively in
   Polars (9s, no vocabulary matrix). It independently rediscovered the same 52
   noise tokens covering 92.2% of documents that our hand-written stopword list
   already removed. Useful as validation; added no recall.
2. **Intersected postal key** — *failed, and was not shipped.* The strict
   5–6 digit regex matched only **7.7%** of S2/S3 rows, because real codes are
   split by punctuation (`KH NO. -570/13`, `G-3/571`). It covered 6.8% of S1
   entities and scored **1.92% entity recall / 0.1077 ceiling** — unusable as a
   primary key. The shipped `k2` uses a more forgiving address-digit prefix
   instead.
3. **Hard-cap at 1,000** — *works, but not for the reason usually claimed.*
   A `.head(1000)` inside `agg` does **not** prevent OOM: the join executes
   first, so the full row set materialises and peak memory is already spent.
   The cap bounds *output* size, not peak memory. We additionally precompute
   per-key bucket sizes once and skip oversized keys, which is what actually
   fixed the tail risk.

## Model Architecture and Feature Engineering

Binary classification with **LightGBM** (~1.13M training rows).

**The calibration fix (V2 model) — our most significant finding.** Our first
model scored 0.9319 macro F₀.₅, but that number was invalid. It was trained on
~40 sampled candidates/S1 while inference supplies up to 1,000, so the
classifier had never seen a low-ranked candidate. In a smoke test it emitted
**89.4 matches per entity** against an expected ~3.7. Retraining with 200 hard
+ 200 random negatives per entity — matching inference density — produced the
honest **0.8517**, emitting 4.66 matches/S1 and 6.2% singletons, closely
tracking the training distribution (3.67 matches, 5.6% singletons).

*The 0.93 → 0.85 drop was not a regression; 0.85 is the real number for the
harder task inference actually faces.* The general lesson: a validation score
is only meaningful at the candidate density the model will meet at inference.

**Feature engineering.** Feature importance confirmed that **address features
carry 3 of the top 4 signals**, compensating for high variance in business-name
transliteration:

- **Address:** `addr_token_set_ratio`, `addr_jaccard_3gram`, `addr_num_overlap`,
  and an `addr_s23_missing` flag for the asymmetric 3.3% missingness in S2/S3
  (S1 addresses were 100% populated in a 300k sample).
- **Name:** `name_jaro_winkler` (rewards prefix matches), `name_levenshtein_ratio`,
  `name_token_sort_ratio`, `name_jac3`, `name_len_diff`.
- **Sanity check:** `country_same` received zero importance, as expected under
  a country hard block.

Threshold **0.64**, the empirical macro-F₀.₅ peak on a 20,000-entity holdout.
The curve is single-peaked and flat-topped, so the optimum is not a knife-edge
artifact.

## Any Other Relevant Information

**Inference optimisation.** Running 1.73M test entities locally, the naive
per-chunk key unpivot of the 10.3M-row S2/S3 table implied **40+ hours**. We
precomputed a global S2/S3 key table once (~10 min, 686 MB) and switched the
per-chunk `is_in` scans to hash semi-joins, cutting this to **~14.5 hours**.

**Engineering pitfalls we hit and documented** (all in `code/README.md`):
Polars `group_by` iteration yields **tuple** keys `('S1-123',)`, not strings —
misusing them silently produced empty output; `u32` `sum()` wraps silently on
large counts (an 11.8-trillion pair count printed as 2.7 billion); `fill_null(0)`
is required before averaging candidate counts or the mean looks artificially
good; and document frequency must count **documents**, not token slots, or it
inflates past 100%.

**Known limitations.** (1) Precision is measured against *sampled* negatives,
so the true ~4,000-candidate pool will yield somewhat lower precision.
(2) The shipped pool is the equality union (95.09%), not the soft+postal pool
(95.48%); since address features dominate, the soft pool should score slightly
higher and is the clearest future gain. (3) `hnswlib`/`faiss` have no Python
3.13 wheels, so soft blocking was evaluated with exact sparse cosine — an
**upper bound** on ANN performance. (4) Validation figures come from 300–20,000
entity samples (±2–4pp).

| soft TF-IDF (name+addr) ∪ postal | 95.48% | 80.4% | 0.9861 | 25,684 |

We also tested and **rejected** a top-K pruner: keeping the top 30 of ~19,244
candidates caps achievable F₀.₅ at 0.83 *regardless of classifier quality*,
because ~3.67 true matches sit below thousands of same-prefix businesses.
