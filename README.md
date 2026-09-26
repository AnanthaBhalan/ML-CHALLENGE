# ML Challenge 2026 — Business Entity Resolution

Linking Source 1 (deduplicated reference) records to their matches in
Sources 2 and 3, across US / India / **France** (test-only country).

## Approach

Three empirically-driven stages. Every design choice below is backed by a
measurement in the `*.txt` result files next to each script.

### 1. Blocking — equality keys (`blocking_eda.py`, `diag_blocking.py`)

Business names are Zipfian, so discrete token buckets trade recall against
candidate volume along a single curve; no setting satisfies both.

| Key | pair recall | entity recall | F0.5 ceiling | cand/S1 |
|---|---|---|---|---|
| country only | 100% | 100% | 1.0000 | 5,365,034 |
| clean_p4 | 83.2% | 59.8% | 0.9273 | 7,744 |
| clean_full | 47.2% | 12.5% | 0.7099 | 77 |
| **UNION(clean_p4, clean_full, postal)** | **95.1%** | **86.3%** | **0.9811** | **19,244** |

`diag_blocking.py` proved the *cause*: 148,707 distinct keys, but the top 500
buckets hold 50% of records, dominated by legal-form/honorific first tokens
(`us|the` 91k, `india|private` 87k, `us|llc` 82k, `india|m` 55k from "M/s").
Stripping them raised recall 77.1%→83.2% but did **not** reduce volume
(7,410→7,744) — the noise tokens were a symptom of low prefix entropy, not
the cause.

### 2. Soft blocking (`soft_block_proto.py`) — evaluated, not shipped

Char n-gram (3,5) cosine, per-country hard block, manual sublinear TF.
hnswlib/faiss have no Python 3.13 wheels here, so exact sparse cosine was
used — an *upper* bound on any ANN index.

| Config | pair recall | F0.5 ceiling |
|---|---|---|
| name only, k=100 | 58.8% | 0.7672 |
| name+address, k=100 | 72.8% | 0.8829 |
| name+address, k=300 | 73.1% | 0.8848 |
| name+address ∪ postal | 95.5% | 0.9861 |

**Negative result, kept deliberately:** soft blocking alone does *not* beat
the equality curve — k=100→300 buys only +0.4pp. Address text is the single
biggest lever (+14pp). hnswlib/faiss remain a pure speed optimisation.

### 3. Pair classifier (`clf_pipeline.py`, `clf_infer.py`)

LightGBM over 11 features (rapidfuzz + Polars). **Address features carry 3
of the top 4 signals** (`addr_token_set`, `addr_jac3`, `name_jac3`), which
is why the address-aware pool was chosen.

Shipped model: `clf_model_v2.txt`, threshold **0.64**, cap 1000.
Validation macro F0.5 = **0.8517** on 8,000 held-out S1 entities.

**Critical calibration finding:** a first model scored 0.9319 but was trained
with only 40 negatives/S1 while inference feeds 1,000 candidates — it emitted
**89 matches/S1 vs 3.7 expected**. Retraining at 200 hard + 200 random
negatives (202 rows/S1) matches inference scale; a 5k test slice then gives
4.66 matches/S1 and 6.2% singletons, consistent with train (3.67 / 5.6%).

## Reproducing

```bash
pip install polars lightgbm rapidfuzz scikit-learn

# 1. train pool + model (cached to parquet)
python clf_pipeline.py --n-train 20000 --n-val 8000 --cap-pool 1000 \
                       --n-hard 200 --n-rand 200 --cache clf_pool2.parquet
CLF_TRAIN_ONLY=1 CLF_CACHE=clf_pool2.parquet python clf_pipeline.py

# 2. test inference -> output/*.tsv
python clf_infer.py --model clf_model_v2.txt --thr 0.64 --chunk 2500

# 3. validate before submitting
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Files

| File | Purpose |
|---|---|
| `blocking_eda.py` | blocking-key sweep (recall / candidates / F0.5 ceiling) |
| `diag_blocking.py` | bucket-distribution diagnosis |
| `soft_block_proto.py` | char n-gram soft blocking evaluation |
| `volume_crusher.py` | exact-IDF + intersected postal key evaluation |
| `prune_eval.py` | top-K pruner (capacity- vs feature-limited) |
| `clf_pipeline.py` | training pool, features, training, threshold sweep |
| `clf_infer.py` | chunked test inference → submission TSVs |
| `*.txt` | raw measurement output backing every number above |

## Notes

- **No external data** is used anywhere (no geocoding, registries, or ER APIs).
- `output/` and `*.parquet` are gitignored: the submission TSVs are
  generated artifacts, and the caches are multi-GB. Regenerate via the
  commands above.
- The model is LightGBM (MIT), well within the ≤8B parameter constraint.
