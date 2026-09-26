# Business Entity Resolution — Reproducible Pipeline

Links each Source 1 business to its matching records in Sources 2 and 3.

**Approach:** deterministic multi-key blocking (high recall) → per-pair feature
extraction → LightGBM binary classifier → threshold tuned for **macro F₀.₅**.

**Validation macro F₀.₅: 0.8517** at threshold 0.64 (20,000 held-out S1 entities).

---

## ⚠️ Model version — read this first

Two model files exist. **Only `clf_model_v2.txt` is valid for submission.**

| Model | Trained on | Val macro F₀.₅ | Status |
|---|---|---|---|
| `clf_model.txt` (v1) | 40 negatives/S1 | 0.9319 *(misleading)* | ❌ **DO NOT USE** |
| `clf_model_v2.txt` (v2) | 200 hard + 200 random negatives/S1 | 0.8517 | ✅ **shipped** |

**Why v1 fails at inference.** v1 was trained and scored on a pool of ~40
candidates per S1 entity, but inference feeds up to 1,000. A classifier never
shown a candidate beyond rank 40 scores almost everything above threshold —
in a smoke test v1 emitted **89.4 matches/S1** against an expected ~3.7.

v2 matches the training and inference distributions. On a 5,000-entity test
slice v2 emits **4.66 matches/S1, 6.2% singletons**, closely tracking the
training distribution (3.67 matches, 5.6% singletons).

The 0.9319 → 0.8517 drop is **not a regression**; 0.8517 is the honest number
for the harder ~200-candidate task that inference actually faces.

---

## Layout

```
src/
  clf_pipeline.py     Stage 1 — candidate pool + labelling + feature matrix + train + threshold sweep
  clf_infer.py        Stage 2 — chunked test inference → output/*.tsv
  blocking_eda.py     Research tool — measures recall ceiling / candidate volume per blocking key
  validate_submission.py  Format checker for the output TSVs
```

The four research scripts (`diag_blocking.py`, `prune_eval.py`,
`soft_block_proto.py`, `volume_crusher.py`) are documented in the methodology
doc; they are not needed to reproduce the submission.

## Reproduce

```bash
pip install -r requirements.txt

# Stage 1 — build pool, train model, sweep threshold
python src/clf_pipeline.py --train-dir <path>/dataset/train
# writes clf_pool.parquet, clf_model.txt (v1), clf_model_v2.txt, threshold sweep log

# Stage 2 — chunked test inference
python src/clf_infer.py --test-dir <path>/dataset/test \
    --index test_s23_keys.parquet --model clf_model_v2.txt \
    --thr 0.64 --chunk 50000 --cap 1000
# writes output/matching_results.tsv and output/candidate_pairs.tsv

# Validate before submitting
python src/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv
```

Useful inference flags: `--limit N` (smoke test on N rows), `--rebuild-index`
(regenerate the S2/S3 key index), `--max-bucket N` (skip blocking keys with
buckets larger than N; default 100,000).

## Runtime / resources

Built and run on a **16 GB Windows machine, Python 3.13, no GPU**.

| Stage | Time |
|---|---|
| Index build (S2/S3 keys → parquet) | ~10 min, 686 MB |
| Training pool (70k S1 entities) | ~33 min |
| Model fit + threshold sweep | ~2 min |
| Test inference (1.73M S1) | **~14.5 h** |

Inference is CPU-bound and single-process. **Omit `n_jobs=-1` when running
inference** so it does not starve other processes.

## Engineering notes (bugs worth not repeating)

1. **Polars `group_by` iteration yields tuple keys** — `('S1-123',)`, not
   `'S1-123'`. Using them directly as dict keys silently produces empty output.
2. **`.head(N)` inside `agg` does not prevent OOM.** The join executes first, so
   the full row set materialises and peak memory is already spent. Count per
   key *before* joining instead — O(records) not O(pairs).
3. **Match training negative count to inference candidate count**, or the
   classifier will over-fire (see the v1/v2 note above).
4. **Explicit `UInt64` casts on `sum()`** — `u32` wraps silently on large
   counts (a 11.8-trillion pair count printed as 2.7 billion).
5. **`.fill_null(0)` before averaging candidate counts**, else entities with
   zero candidates are dropped and the mean looks artificially good.
6. **Count documents, not token slots, for IDF** — `pl.len()` after `explode`
   counts occurrences, inflating document frequency above 100%.
