"""
STEP 0 GATE: measure what --cap 300 costs vs --cap 1000.

Runs the SHIPPED inference path (clf_infer.process_chunk + clf_model_v2) over a
small sample of TRAIN S1 entities, scoring against train_ground_truth.tsv.
Reports macro F0.5 at each cap so we can decide whether the speed is worth it.

This touches only the training data -- the live test run is not affected.

    python cap_eval.py --sample 1500
"""

import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import clf_infer as CI
from clf_infer import add_keys, norm, process_chunk, FEATURES

TRAIN = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN)
    ap.add_argument("--model", default="clf_model_v2.txt")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--thr", type=float, default=0.64)
    ap.add_argument("--max-bucket", type=int, default=100_000)
    ap.add_argument("--caps", default="1000,300")
    args = ap.parse_args()
    t0 = time.time()

    print("loading train s1 + s23 ...", flush=True)
    s1 = add_keys(norm(pl.scan_csv(os.path.join(args.train_dir, "train_source1.tsv"),
                                   separator="\t"))).collect()
    s23 = add_keys(norm(pl.concat([
        pl.scan_csv(os.path.join(args.train_dir, "train_source2.tsv"), separator="\t"),
        pl.scan_csv(os.path.join(args.train_dir, "train_source3.tsv"), separator="\t"),
    ], how="vertical"))).collect()
    print(f"  s1={s1.height:,} s23={s23.height:,} ({time.time()-t0:.0f}s)", flush=True)

    gt = (pl.scan_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), separator="\t")
          .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")])
          .explode("m", empty_as_null=True)
          .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
          .select(["source1_entity_id", pl.col("m").alias("mid")]).collect())

    # Map S1 id -> 0..n-1 so macro_f05 can index a dense range.
    s1 = s1.sample(fraction=1.0, shuffle=True, seed=42) \
            .head(args.sample).with_row_index("_rid")
    ids = s1["entity_id"].to_list()
    id2rid = {e: i for i, e in enumerate(ids)}

    truth = [(id2rid[a], b) for a, b in
             gt.select(["source1_entity_id", "mid"]).iter_rows()
             if a in id2rid]
    print(f"sample entities={len(ids):,}  true pairs={len(truth):,}", flush=True)

    bsz = CI.build_bucket_sizes(s23, args.max_bucket)
    clf = lgb.Booster(model_file=args.model)

    from clf_pipeline import macro_f05

    results = {}
    for cap in [int(x) for x in args.caps.split(",")]:
        tc = time.time()
        pred, ncand = [], []
        n_sing = 0
        for part in s1.iter_slices(args.chunk):
            s1ids, cand_map, match_map = process_chunk(
                part, s23, clf, cap, args.thr,
                bucket_sizes=bsz, max_bucket=args.max_bucket)
            for s1id in s1ids:
                rid = id2rid[s1id]
                for c in match_map.get(s1id, []):
                    pred.append((rid, c))
                ncand.append(len(cand_map.get(s1id, [])))
                if not match_map.get(s1id):
                    n_sing += 1
        f05 = macro_f05(truth, pred, len(ids))
        results[cap] = f05
        print(f"\n[cap {cap}] macro F0.5 = {f05:.4f}  "
              f"pred_pairs={len(pred):,}  avg_cand={np.mean(ncand):.0f}  "
              f"singletons={n_sing/len(ids)*100:.1f}%  ({time.time()-tc:.0f}s)", flush=True)

    caps = [int(x) for x in args.caps.split(",")]
    if len(caps) == 2:
        hi, lo = caps
        d = results[lo] - results[hi]          # positive => lower cap scored BETTER
        print(f"\n=== cap {lo} vs cap {hi}: {d:+.4f} ===")
        if d >= -0.03:
            print(f"  ACCEPTABLE - cap {lo} costs <= 0.03 (it scored BETTER here)")
        else:
            print(f"  TOO COSTLY - cap {lo} loses {abs(d):.4f}; keep the full run")


if __name__ == "__main__":
    main()
