# Handoff: finishing test inference on a bigger machine

**Why you're reading this:** the 2026-09-26 run on a 15 GB / 8-core Ryzen
finished only a fraction of the test set in the available time. The pipeline is
validated and correct; it just needs more compute than that laptop could give.

**Time required on your machine:** 1–2 hours with ≥32 GB RAM, or ~4 h with
16 GB. You do **not** need a GPU.

---

## TL;DR (Linux/macOS)

```bash
pip install -r student_resource/requirements.txt

cd student_resource

# 1) Build the S2/S3 key index ONCE (~10 min, ~690 MB)
python clf_infer.py --rebuild-index --limit 0   # stops early; just builds index

# 2) Run N shards in parallel. N = 3 is safe on 32 GB; N = 2 on 16 GB.
#    Do NOT use N=3 on 16 GB -- it OOMs.
TOTAL=$(wc -l < dataset/test/test_source1.tsv)   # 1,732,545 lines incl. header
TOTAL=$((TOTAL - 1))                             # 1,732,544 data rows

N=3
STEP=$((TOTAL / N))
for i in $(seq 0 $((N-1))); do
  START=$((i * STEP))
  if [ "$i" -eq $((N-1)) ]; then END=$TOTAL; else END=$(((i+1) * STEP)); fi
  python clf_infer.py \
      --model clf_model_v2.txt --thr 0.64 --cap 1000 --chunk 2500 \
      --shard-start $START --shard-end $END --out-suffix "_s$i" \
      > shard$i.log 2>&1 &
  sleep 30          # stagger: stops all shards loading the index at once
done
wait

# 3) Merge + verify (aborts on duplicate or missing S1 id)
python merge_shards.py --shards _s0,_s1,_s2 --total 1732544

# 4) Official format check -- MUST exit 0
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv
```

Windows (PowerShell) equivalent of step 2:

```powershell
$Total = 1732544; $N = 3; $Step = [math]::Floor($Total / $N)
for ($i = 0; $i -lt $N; $i++) {
  $S = $i * $Step
  $E = if ($i -eq $N-1) { $Total } else { ($i+1) * $Step }
  Start-Process python -ArgumentList 'clf_infer.py','--model','clf_model_v2.txt',
    '--thr','0.64','--cap','1000','--chunk','2500',
    '--shard-start',"$S",'--shard-end',"$E",'--out-suffix',"_s$i" `
    -RedirectStandardOutput "shard$i.log" -RedirectStandardError "shard$i.err" `
    -WindowStyle Hidden
  Start-Sleep -Seconds 30
}
```

## Optional: reuse the 104,919 rows already computed

`student_resource/output/*_base.tsv` contains **104,919 already-finished rows**
(rows 0 – 104,918 of `test_source1.tsv`), produced by the same model, cap and
threshold. They are internally consistent: identical ID order in both files,
6.17% singletons, 104,919 unique S1 IDs.

To skip that work, start your shards at row **104,919** and pass `base` first to
the merge:

```bash
START0=104919
N=3
SPAN=$(( (TOTAL - START0) / N ))
for i in $(seq 0 $((N-1))); do
  START=$((START0 + i * SPAN))
  if [ "$i" -eq $((N-1)) ]; then END=$TOTAL; else END=$((START0 + (i+1) * SPAN)); fi
  python clf_infer.py --model clf_model_v2.txt --thr 0.64 --cap 1000 \
      --chunk 2500 --shard-start $START --shard-end $END --out-suffix "_s$i" \
      > shard$i.log 2>&1 &
  sleep 30
done
wait

python merge_shards.py --shards base,_s0,_s1,_s2 --total 1732544
```

**Saves ~6% of the runtime.** If you would rather not trust the pre-computed
rows, ignore this section and run the TL;DR block from row 0 — it produces a
complete, self-consistent result on its own.

---

## Rules that MUST be followed

| Rule | Why |
|---|---|
| **Use `clf_model_v2.txt`, never `clf_model.txt`** | v1 was trained at 40 negatives/entity but inference supplies 1000. It over-fires at **89 matches/S1** vs 3.7 expected. |
| **Keep `--cap 1000`** | Measured: 1000 → 6.2% singletons / 4.5 pairs per S1, matching the training distribution. A cap of 300 pushed singletons to 16% and halved the match count. |
| **Keep `--thr 0.64`** | Empirical macro-F₀.₅ optimum on the 20k-entity holdout. |
| **Shard ranges must not overlap** | Ranges are row offsets into `test_source1.tsv`. Overlap ⇒ duplicate S1 ids ⇒ `merge_shards.py` aborts. |
| **Merge in shard order `_s0,_s1,_s2`** | The merged file is expected in test-source file order. |
| **Let `validate_submission.py` exit 0** | Tab-separated, exact headers, empty 2nd column for singletons, every matched id a subset of the candidate ids. |

## Sanity numbers (check yours match)

```
singletons            ~6.2%      (train ground truth: 5.58%)
matched pairs per S1  ~4.5-4.9  (train ground truth: 3.67)
rows in each TSV       1,732,544 data rows (plus 1 header)
candidate_pairs.tsv   ~1.2-1.3 GB
```

If singletons land far outside **4–9%**, something is wrong — stop and check
the flags before packaging.

## What NOT to bother re-running

Training is done. `clf_model_v2.txt` (4 MB) and the threshold sweep are
committed. You do **not** need `clf_pipeline.py`, the 686 MB key index, or the
2 GB dataset — get the dataset from the competition portal and build the index
locally (step 1).

## Files that matter

```
student_resource/clf_infer.py        <- entry point (has --shard-* flags)
student_resource/merge_shards.py     <- merge + duplicate/missing-id guard
student_resource/utils/validate_submission.py
student_resource/clf_model_v2.txt    <- THE model
```

## Packing the submission

From the repo root, `python make_submission_zip.py` verifies row counts, runs
the official validator, copies outputs into `submission_package/output/`, and
writes `submission.zip`. It refuses to build a partial or malformed package.
