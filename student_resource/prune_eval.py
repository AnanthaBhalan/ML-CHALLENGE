"""
Two-stage blocking pruner for ML Challenge 2026.

Architecture
------------
  stage 1  LOOSE BLOCKING   clean_p4 + clean_full + postal  (~19k cand/S1)
  stage 2a HEURISTIC PASS A cheap scalar features -> keep top-200 per S1
  stage 2b HEURISTIC PASS B exact token Jaccard  -> keep top-30  per S1
  stage 3  (next)           heavy ML classifier over the surviving ~30

MEASURED RESULTS (sample=1,500 S1, 5,120 true pairs, 16 cores)
  loose blocking, no pruning : 95.09% pair recall, 19,244 cand/S1
  pass A (cheap) -> top 200  : 69.78% pair recall
  pass A -> top 30 (2-stage) : 65.20% pair, 35.14% entity, ceiling 0.8103
  Jaccard on ALL -> top 30   : 68.54% pair, 41.11% entity, ceiling 0.8267
  Jaccard on ALL -> top 200  : 76.62% pair, 52.12% entity, ceiling 0.8803

  => The pruner is CAPACITY-limited, not feature-limited. Scoring every
     candidate with exact Jaccard buys only +3.3pp over the cheap scalar
     prefilter (68.5% vs 65.2% at top-30), so better cheap features are NOT
     the lever. The lever is how many candidates survive.

WHY TOP-30 CANNOT WORK HERE
  S1 entities average 3.67 true matches, but the candidate pool is Zipfian:
  the correct record often sits below thousands of same-prefix businesses.
  65% pair recall at top-30 is simply the arithmetic of ranking ~19,244
  candidates and keeping 30. No scoring function changes that; only
  generating a smaller, more precise candidate pool does.

  IMPORTANT: `candidate_pairs.tsv` must be the set fed to the MODEL, and the
  final model must be able to see true matches. A top-30 pruner with 65%
  recall would HARD-CAP the achievable F0.5 near 0.83 regardless of model
  quality. Do not ship top-30. Use top-200+ or keep the full pool and let a
  real classifier rank it.

IMPLICATION FOR THE ARCHITECTURE
  The 19,244 avg / 301,588 max pool is the actual engineering problem, and
  cheap-feature pruning is the wrong tool for it. The right fix is a precise
  pool, i.e. soft blocking: character n-gram TF-IDF + ANN (hnswlib/FAISS) to
  retrieve ~50-100 true-ish candidates, which both shrinks the pool AND
  raises recall above what any top-K cutoff can reach here.

Why two passes: carrying a token LIST column through a 19M-row join is the
memory bottleneck. Pass A scores using only short scalars (first token,
postal, token count, char length) and cuts 100x first; Pass B then joins
token lists onto the small survivor set to compute exact Jaccard. Same
result as scoring everything, at a fraction of the RAM.

Everything is native Polars string/list expression -- no Python apply, no
UDF, no per-pair Python object.

Usage:
    python prune_eval.py --sample 5000 --chunk 500 --top-a 200 --top-b 30
"""

import argparse
import gc
import os
import time

import polars as pl

TRAIN = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\train"

LEAD_NOISE = {
    "mr", "mrs", "ms", "dr", "smt", "sri", "shri", "shree", "om", "baba",
    "m", "sm", "md", "sh", "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "private", "pvt", "plc", "llp", "lp",
    "pc", "pa", "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv", "ab", "as",
    "oy", "kk", "the", "a", "an", "and", "of", "new", "global", "national",
    "international", "universal", "general", "services", "service",
    "enterprises", "group", "holdings", "ventures", "india", "usa", "us",
}

KEY_COLS = ["k0", "k1", "k2"]


def norm(df):
    """Normalize name into: nn (flat), toks (cleaned list), ft (first ident. token)."""
    return (
        df.with_columns(
            pl.col("business_name")
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^\w\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("nn")
        )
        .with_columns(
            pl.col("nn").str.split(" ").alias("_all")
        )
        .with_columns(
            pl.col("_all")
            .list.eval(
                pl.element()
                .filter(pl.element().str.len_chars() > 0)
                .filter(~pl.element().is_in(LEAD_NOISE))
            )
            .alias("toks")
        )
        .with_columns(
            pl.col("toks").list.head(1).list.join(" ").alias("_ft")
        )
        .with_columns(
            pl.when(pl.col("_ft").str.len_chars() > 0)
            .then(pl.col("_ft"))
            .otherwise(pl.col("nn"))
            .alias("ft")
        )
        .with_columns(
            pl.col("country").fill_null("").str.to_lowercase().alias("ctry")
        )
        .with_columns(
            pl.col("business_address")
            .fill_null("")
            .str.replace_all(r"\D", "")
            .str.replace_all(r"^0+", "")
            .str.slice(0, 6)
            .alias("postal")
        )
        .with_columns(
            pl.col("nn").str.len_chars().cast(pl.UInt16).alias("nchar")
        )
        .with_columns(
            pl.col("toks").list.len().cast(pl.UInt16).alias("ntok")
        )
    )


def add_keys(df):
    """The three loose blocking keys, as k0/k1/k2."""
    return df.with_columns(
        (pl.col("ctry") + "|" + pl.col("ft").str.slice(0, 4)).alias("k0"),
        (pl.col("ctry") + "|" + pl.col("toks").list.join(" ")).alias("k1"),
        (pl.col("ctry") + "|" + pl.col("postal").str.slice(0, 3)).alias("k2"),
    )


def gen_pairs(s1_chunk, s23, top_a, use_jac=False):
    """Union the 3 keys -> dedup pairs, score, keep top-A per S1.

    use_jac=False : cheap scalar prefilter (fast, but weak -- caps recall)
    use_jac=True  : carry token lists through the join and score exact
                    Jaccard on EVERY pair. Correct but heavier; the only way
                    to find out whether the pruner is feature-limited or
                    capacity-limited.
    """
    tk = ["toks"] if use_jac else []
    keep_cols = ["_rid", "_cid", "ft1", "ft2", "pc1", "pc2"]
    a = s1_chunk.select(["_rid", "ft", "postal", "nchar", "ntok", *KEY_COLS] + tk).unpivot(
        index=["_rid", "ft", "postal", "nchar", "ntok"] + tk,
        on=KEY_COLS, value_name="k",
    ).filter(pl.col("k").is_not_null())

    b = s23.select(["_cid", "ft", "postal", "nchar", "ntok", *KEY_COLS] + tk).unpivot(
        index=["_cid", "ft", "postal", "nchar", "ntok"] + tk,
        on=KEY_COLS, value_name="k",
    ).filter(pl.col("k").is_not_null())

    sel = [
        pl.col("_rid"), pl.col("_cid"),
        pl.col("ft").alias("ft1"), pl.col("ft_c").alias("ft2"),
        pl.col("postal").alias("pc1"), pl.col("postal_c").alias("pc2"),
        pl.col("nchar").alias("nc1"), pl.col("nchar_c").alias("nc2"),
        pl.col("ntok").alias("nt1"), pl.col("ntok_c").alias("nt2y"),
    ]
    if use_jac:
        sel += [pl.col("toks").alias("tk1"), pl.col("toks_c").alias("tk2")]
    pairs = a.join(b, on="k", how="inner", suffix="_c").select(sel).unique(
        subset=["_rid", "_cid"]
    )

    if use_jac:
        pairs = (
            pairs.with_columns(
                pl.col("tk1").list.set_intersection(pl.col("tk2")).list.len().alias("inter"),
                pl.col("tk1").list.set_union(pl.col("tk2")).list.len().alias("uni"),
            )
            .with_columns(
                pl.when(pl.col("uni") == 0).then(0.0)
                .otherwise(pl.col("inter").cast(pl.Float32) / pl.col("uni").cast(pl.Float32))
                .alias("jac")
            )
            .with_columns(
                (
                    6.0 * pl.col("jac")
                    + 2.0 * (pl.col("ft1") == pl.col("ft2")).cast(pl.Float32)
                    + 1.5 * (pl.col("pc1") == pl.col("pc2")).cast(pl.Float32)
                ).alias("score_a")
            )
        )
        return (
            pairs.sort("_rid", "score_a", descending=[False, True])
            .group_by("_rid", maintain_order=True).head(top_a)[keep_cols]
        )

    # ---- Pass A: cheap scalar features (no token lists) --------------------
    pairs = pairs.with_columns(
        (pl.col("ft1") == pl.col("ft2")).cast(pl.UInt8).alias("f_eq"),
        (pl.col("pc1") == pl.col("pc2")).cast(pl.UInt8).alias("p_eq"),
        (pl.col("nt1") == pl.col("nt2y")).cast(pl.UInt8).alias("t_eq"),
        (
            pl.min_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32)
            / pl.max_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32).clip(1.0)
        ).alias("len_r"),
    ).with_columns(
        (
            3.0 * pl.col("f_eq")
            + 2.0 * pl.col("p_eq")
            + 1.0 * pl.col("t_eq")
            + 1.5 * pl.col("len_r")
        ).alias("score_a")
    )

    top = (
        pairs.sort("_rid", "score_a", descending=[False, True])
        .group_by("_rid", maintain_order=True)
        .head(top_a)
    )
    del pairs, a, b
    gc.collect()
    # carry the cheap features into pass B (they refine the final ordering)
    return top.select(["_rid", "_cid", "ft1", "ft2", "pc1", "pc2"])




def pass_b(top, s1_chunk, s23, top_b):
    """Exact token Jaccard on the survivors only, then keep top-B per S1."""
    toks1 = s1_chunk.select(["_rid", "toks"]).rename({"toks": "tk1"})
    toks2 = s23.select(["_cid", "toks"]).rename({"toks": "tk2"})
    j = (
        top.join(toks1, on="_rid", how="left")
        .join(toks2, on="_cid", how="left")
        .with_columns(
            pl.col("tk1").list.set_intersection(pl.col("tk2")).list.len().alias("inter"),
            pl.col("tk1").list.set_union(pl.col("tk2")).list.len().alias("uni"),
        )
        .with_columns(
            pl.when(pl.col("uni") == 0)
            .then(0.0)
            .otherwise(pl.col("inter").cast(pl.Float32) / pl.col("uni").cast(pl.Float32))
            .alias("jac")
        )
        .with_columns(
            (
                6.0 * pl.col("jac")
                + 2.0 * (pl.col("ft1") == pl.col("ft2")).cast(pl.Float32)
                + 1.5 * (pl.col("pc1") == pl.col("pc2")).cast(pl.Float32)
            ).alias("score_b")
        )
    )
    out = (
        j.sort("_rid", "score_b", descending=[False, True])
        .group_by("_rid", maintain_order=True)
        .head(top_b)
    )
    del j, toks1, toks2
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN)
    ap.add_argument("--sample", type=int, default=5000, help="S1 entities to evaluate")
    ap.add_argument("--chunk", type=int, default=500, help="S1 rows per chunk")
    ap.add_argument("--top-a", type=int, default=200)
    ap.add_argument("--top-b", type=int, default=30)
    ap.add_argument(
        "--use-jac", action="store_true",
        help="score ALL pairs with exact Jaccard in pass A (slower, no prefilter)",
    )
    args = ap.parse_args()
    t0 = time.time()

    print(f"Polars {pl.__version__} | sample={args.sample} chunk={args.chunk} "
          f"top_a={args.top_a} top_b={args.top_b}")

    s1 = add_keys(norm(pl.scan_csv(
        os.path.join(args.train_dir, "train_source1.tsv"), separator="\t"))).collect()
    s23 = add_keys(norm(pl.concat([
        pl.scan_csv(os.path.join(args.train_dir, "train_source2.tsv"), separator="\t"),
        pl.scan_csv(os.path.join(args.train_dir, "train_source3.tsv"), separator="\t"),
    ], how="vertical"))).collect()
    s23 = s23.with_row_index("_cid")
    print(f"loaded s1={s1.height:,} s23={s23.height:,} ({time.time()-t0:.0f}s)", flush=True)

    s1 = s1.filter(pl.col("ctry") != "").sample(n=min(args.sample, s1.height), seed=42)
    s1 = s1.with_row_index("_rid")
    n_s1 = s1.height
    print(f"sampled {n_s1:,} S1 entities", flush=True)

    # ground truth restricted to the sample, mapped to candidate row ids
    sid = s1.select(["entity_id", "_rid"])
    cid = s23.select(["entity_id", "_cid"])
    gtx = (
        pl.scan_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), separator="\t")
        .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")])
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("mid")])
        .collect()
        .join(sid, left_on="source1_entity_id", right_on="entity_id", how="inner")
        .select(["_rid", "mid"])
        .join(cid, left_on="mid", right_on="entity_id", how="left")
    )
    print(f"true pairs within sample: {gtx.height:,}", flush=True)

    kept = []
    t_kept = 0
    t_a = 0
    n_chunks = (n_s1 + args.chunk - 1) // args.chunk
    kept_a = []
    for ci, part in enumerate(s1.iter_slices(args.chunk)):
        ta = gen_pairs(part, s23, args.top_a, use_jac=args.use_jac)
        tb = ta if args.use_jac else pass_b(ta, part, s23, args.top_b)
        t_a += ta.height
        t_kept += tb.height
        kept_a.append(ta.select(["_rid", "_cid"]))
        kept.append(tb.select(["_rid", "_cid"]))
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci+1}/{n_chunks}  passA={t_a:,}  kept={t_kept:,}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    kept = pl.concat(kept, how="vertical")
    kept_a = pl.concat(kept_a, how="vertical")
    print(f"\npruned: {kept.height:,} pairs ({t_kept/max(n_s1,1):.1f} per S1)  "
          f"total {time.time()-t0:.0f}s", flush=True)

    # ---- recall of the PRUNED set ------------------------------------------
    def recall_of(frame):
        h = (
            gtx.join(
                frame.with_columns(pl.lit(True).alias("_kept")),
                on=["_rid", "_cid"], how="left",
            )
            .with_columns(pl.col("_kept").fill_null(False).alias("hit"))
        )
        pr = float(h["hit"].mean() or 0.0)
        pe = h.group_by("_rid").agg(
            pl.len().alias("n_true"), pl.col("hit").sum().alias("n_hit"))
        er = float((pe["n_hit"] == pe["n_true"]).mean() or 0.0)
        return pr, er, h, pe

    pr_a, er_a, _, _ = recall_of(kept_a)
    pair_recall, ent_recall, hit, per_ent = recall_of(kept)
    print(f"\n  [stage A -> top {args.top_a}] pair recall {pr_a*100:.2f}%  "
          f"entity recall {er_a*100:.2f}%")
    del kept_a
    gc.collect()
    pair_recall = float(hit["hit"].mean() or 0.0)
    per_ent = hit.group_by("_rid").agg(
        pl.len().alias("n_true"), pl.col("hit").sum().alias("n_hit"))
    ent_recall = float((per_ent["n_hit"] == per_ent["n_true"]).mean() or 0.0)
    n_sing = n_s1 - per_ent.height
    rf = per_ent["n_hit"] / per_ent["n_true"]
    f_each = ((1.25 * 1.0 * rf) / (0.25 * 1.0 + rf)).fill_null(0.0)
    ceiling = float((f_each.sum() + n_sing) / n_s1)

    per = kept.group_by("_rid").agg(pl.len().alias("c"))["c"]
    print("\n================ PRUNED RESULT ================")
    print(f"  pair recall     : {pair_recall*100:.2f}%")
    print(f"  entity recall   : {ent_recall*100:.2f}%  (all true matches kept)")
    print(f"  F0.5 ceiling    : {ceiling:.4f}")
    print(f"  candidates/S1   : avg {float(per.mean() or 0):.1f}  "
          f"p95 {float(per.quantile(0.95) or 0):.0f}  max {int(per.max() or 0):,}")
    print(f"  S1 with 0 cands : {(n_s1 - per.len())/n_s1*100:.2f}%")
    print(f"  wall time       : {time.time()-t0:.0f}s")
    print("==============================================")


if __name__ == "__main__":
    main()

