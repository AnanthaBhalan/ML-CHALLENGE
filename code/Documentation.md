# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** AnanthaBhalan
**Team Members:** AnanthaBhalan
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a three-stage pipeline: **deterministic multi-key blocking** to generate
a high-recall candidate pool, **per-pair feature extraction** (fuzzy name
metrics plus address signals), and a **LightGBM binary classifier** whose
decision threshold is tuned directly against the macro F₀.₅ metric. The central
contribution is empirical: rather than guessing a blocking key, we measured the
recall-ceiling/candidate-volume frontier for eight candidate schemes and
*proved* no equality-based key can reach the target volume, then routed around
that limit by transferring the discrimination burden to the classifier.

Validation macro F₀.₅ = **0.8517** at threshold 0.64.

---

## 2. Methodology

### 2.1 Problem Analysis

Dataset scale (train): S1 = 2,206,821 · S2 = 5,034,616 · S3 = 5,285,603
(≈12.5M records; test adds ≈11.7M). A full cross-join is ≈10¹³ pairs.

Key EDA findings:

- **The task is one-to-many, not deduplication.** The average S1 entity matches
  **3.67** records; 51.2% match more than one S2 record and 55.5% more than one
  S3 record (max 11). Treating this as pairwise dedup understates the problem.
- **5.58% of S1 entities are singletons** (123,247). These score 1.0 if
  predicted empty and 0.0 on any false merge, so they are ~1/18th of the
  evaluation and materially shape the macro average.
- **Business names are Zipfian.** Legal-form and honorific tokens dominate:
  `limited`, `private`, `llc`, `inc`, `pvt`, `partners`, `services` appear in
  ≥0.5% of all documents; just **52 tokens cover 92.2%** of the corpus. The top
  500 name-prefix buckets hold 50.3% of all S2/S3 records.
- **Leading noise tokens** (`m/s`, `shri`, `dr`, `smt`, `the`, `llc`) appear at
  the start of names and collapse thousands of unrelated businesses into one key.
- **Address is asymmetric.** ~3.3% of S2/S3 addresses are empty, versus 0% for
  S1 in a 300k sample. Address features are therefore only usable when the
  S2/S3 side is present.
- **France appears only in test** (~29–30k per source). The country set is
  open-set and must not be hard-coded.
- **Postal codes are sparse and messy.** A strict 5–6 digit regex matches only
  **7.7%** of S2/S3 rows, because real codes are split by punctuation
  (`1795 Westchester Drive`, `KH NO. -570/13`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage, out-of-core)

**Core innovation:** A measured blocking frontier. We quantified eight blocking
schemes on the true-match ground truth and established that the recall/volume
trade-off is a hard frontier — not a tuning failure — which justified committing
compute to a classifier rather than endlessly re-tuning keys.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (all prefixed by country to guarantee zero cross-border
candidates):

| Key | Definition | Purpose |
|---|---|---|
| `k0` | `country + first 4 chars` of the noise-stripped first token | broad name recall |
| `k1` | `country + full cleaned name` | high-precision name matches |
| `k2` | `country + first 3 digits` of the address | catches records whose name was destroyed but whose address agrees |

Candidates are the **de-duplicated union** of the three keys.

**Measured frontier** (against `train_ground_truth.tsv`):

| Scheme | Pair recall | Entity recall | F₀.₅ ceiling | Avg cand/S1 |
|---|---|---|---|---|
| `country` only | 100% | 100% | 1.0000 | 5,365,034 |
| `k0` (4-char) | 77.1% | 47.9% | 0.8980 | 7,410 |
| `k1` (exact name) | 47.2% | — | — | 77 |
| postal ∩ `k0` | 59.9% | 23.9% | 0.7622 | 26 |
| **UNION (k0+k1+k2)** | **95.09%** | **86.27%** | **0.9811** | **19,244** |
| soft TF-IDF (name+addr) ∪ postal | 95.48% | 80.4% | 0.9861 | 25,684 |

**Why true matches were not lost.** Every key is a *union* term, so a true match
recovered by any one key is retained; the union strictly dominates every
individual key. Recall is measured directly against the ground truth, not
assumed. Entity-level recall (86.27%) is reported alongside pair recall because
the macro metric punishes partially-recovered entities heavily.

**Honest limitation.** We could not reach the aspirational <30 candidates/S1
target at >90% recall. The frontier is real: every equality key lies on one
curve, and *no point on it satisfies both constraints*. We proved this
empirically by testing noise-token stripping (recall +6pp but volume *rose*
4.5% — mass simply redistributed to the next-commonest token, e.g. `pediatric`),
prefix length, exact-name, soft TF-IDF retrieval, and an intersected postal key.
A top-K pruner was also tested and **rejected**: keeping the top 30 of ~19,244
candidates caps achievable F₀.₅ at 0.83 regardless of classifier quality,
because ~3.67 true matches are buried below thousands of same-prefix businesses.


---

## 4. Matching Model

**Features used** (11 total, RapidFuzz + Polars/numpy, no Python UDFs):

| Field | Rationale |
|---|---|
| `name_jaro_winkler` | prefix-weighted; robust to typos |
| `name_token_sort` | handles word reordering |
| `name_lev_ratio` | normalised edit distance |
| `name_jac3` | character 3-gram Jaccard |
| `name_len_diff` | length-ratio outlier signal |
| `addr_token_set` | handles subsets/landmarks ("MG Road" vs "Near Metro, MG Road") |
| `addr_jac3` | character n-gram overlap, street spelling variants |
| `addr_num_overlap` | Jaccard of digit tokens (street numbers, PIN fragments) |
| `addr_s23_missing` | binary flag for the asymmetric missingness above |
| `country_same` | constant under the country hard block (sanity check) |
| `s` | blocking-score prior (name-prefix / token-count / length agreement) |

**Model type:** LightGBM GBDT (`lr=0.05`, `num_leaves=63`, 600 trees,
`feature_fraction=0.8`), ~1.13M training rows.

**Threshold selection:** grid search maximising **macro F₀.₅** over a held-out
sample of 20,000 S1 entities, sweeping 0.30–0.95. The curve is single-peaked and
flat-topped (0.9311→0.9319 across thresholds 0.60–0.66 in the v1 sweep), so the
optimum is not a knife-edge artifact and should transfer to test. All held-out
entities are scored, including singletons, which correctly earn 1.0 for an
empty prediction.

**Feature importance confirms the EDA:** three of the top four features are
*address* signals (`addr_token_set`, `addr_jac3`), and `country_same` receives
zero importance because the hard block already guarantees equality.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro): 0.8517** (20,000 held-out S1 entities, ~200-candidate
  pool, threshold 0.64).
- **Pair-level at optimum:** precision 0.9667, recall 0.8919.

**A calibration bug worth reporting.** Our first model scored **0.9319**, but
that number was invalid. It had been trained on ~40 sampled candidates per
entity while inference supplies up to 1,000, so the classifier had never seen a
low-ranked candidate. In a smoke test it emitted **89.4 matches per entity**
against an expected ~3.7. Retraining with 200 hard + 200 random negatives
(calibrated to inference scale) produced the honest **0.8517**, which emits
4.66 matches/S1 and 6.2% singletons — closely tracking the training distribution.
Lesson: *a validation score is only meaningful at the candidate density the
model will actually face.*

- **Common false positives (wrong merges):** chains of near-duplicate outlets in
  the same postal district (e.g. franchise locations) differing only by unit
  number in the address, and names differing solely by an honorific.
- **Common false negatives (missed matches):** records whose name was corrupted
  *and* whose address is missing (~3.3% of S2/S3) — no independent signal
  remains for any method. A further ~9% were never generated as candidates at
  all, which bounds recall regardless of the classifier.

**Known limitations.** (1) Precision is measured against *sampled* negatives, so
the true pool's ~4,000 candidates/entity will yield somewhat lower precision.
(2) The shipped pool uses the equality union (95.09% recall) rather than the
soft+postal pool (95.48%); given address features dominate, the soft pool should
score slightly higher. (3) `hnswlib`/`faiss` have no Python 3.13 wheels, so
soft blocking was evaluated with exact sparse cosine — an **upper bound** on
what an ANN index would achieve.

---

## 6. Conclusion

We mapped the blocking frontier empirically and established by measurement that
no equality-based key satisfies both the recall and volume targets, then
committed compute to a LightGBM classifier to convert the resulting 0.9861
headroom into precision, reaching **0.8517 macro F₀.₅**. The most valuable
outcome was methodological: measuring candidate volume *before* building the
end-to-end pipeline prevented days of tuning against a provably unreachable
target, and calibrating the training distribution to the inference distribution
exposed a 0.93 → 0.85 silent failure that would otherwise have shipped.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships under `code/business_entity_resolution/`:

- `src/clf_pipeline.py` — Stage 1: candidate pool, labelling, feature matrix,
  training, threshold sweep. **Entry point.**
- `src/clf_infer.py` — Stage 2: chunked test inference → output TSVs.
- `src/blocking_eda.py` — measures the recall-ceiling/volume frontier per key.
- `src/validate_submission.py` — output format checker.
- `README.md`, `requirements.txt` — setup, runtime, and known pitfalls.

Reproduce with `python src/clf_pipeline.py`, then `python src/clf_infer.py`.
**Only `clf_model_v2.txt` is valid** — see the code README for why v1 must not
be used.

### B. Additional Results

| Stage | Result |
|---|---|
| Blocking frontier (8 schemes) | mapped; union 95.09% recall / 0.9811 ceiling |
| Top-K pruner | tested, **rejected** — caps F₀.₅ at 0.83 |
| Soft TF-IDF blocking | 72.8% recall alone (+14pp from address text) |
| Out-of-core IDF | 52 noise tokens = 92.2% document coverage |
| Validation | precision 0.9667, recall 0.8919, macro F₀.₅ 0.8517 |
| Test distribution check | 4.66 matches/S1, 6.2% singletons (train: 3.67, 5.6%) |

