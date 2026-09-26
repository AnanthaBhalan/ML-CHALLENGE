"""
Blocking-key EDA for ML Challenge 2026 (Business Entity Resolution).

Measures, for each candidate blocking key:
  * pair-level recall ceiling  (share of train true matches that survive blocking)
  * entity-level recall ceiling (share of S1 entities keeping ALL their true matches)
  * macro-F0.5 ceiling         (the leaderboard metric, upper-bounded by blocking)
  * avg / p95 / max candidates per S1 entity
  * total candidate pairs      (decides whether you OOM)

Design note (why this differs from the naive inner-join version):
Materialising the full candidate-pair frame explodes -- a loose key over
2.2M x 10.3M records produces billions of rows. Instead we:
  (a) count candidates per KEY, then join those counts onto S1 -> O(records)
  (b) for recall, join the exploded ground truth (7.6M rows) against per-entity keys
Both avoid the cross product and give identical numbers, far faster.

Run:
    python blocking_eda.py --train-dir dataset/train --keys all
"""

import argparse
import gc
import os
import time

import polars as pl


def f05(precision, recall):
    if precision <= 0 and recall <= 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


# Leading tokens that carry no identifying signal. Verified against the actual
# first-token frequency table in diag_output.txt (the, private, llc, m, inc,
# dr, sri, mr, shri, smt ...). Removing them collapses the mega-buckets.
LEAD_NOISE = {
    # honorifics / respect prefixes
    "mr", "mrs", "ms", "dr", "smt", "sri", "shri", "shree", "om", "baba",
    "m", "sm", "md", "sh",
    # legal entity-form tokens, both prefix and suffix positions
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "private", "pvt", "plc", "llp", "lp", "pc", "pa",
    "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv", "ab", "as", "oy", "kk",
    "the", "a", "an", "and", "of",
    # generic / promotional
    "new", "global", "national", "international", "universal", "general",
    "services", "service", "enterprises", "group", "holdings", "ventures",
    "india", "usa", "us",
}

# Trailing legal-form tokens: stripped only from the END (LLC Metro -> Metro).
TAIL_NOISE = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "private", "pvt", "plc", "llp", "lp", "pc",
    "gmbh", "sas", "sarl", "bv", "nv", "ag",
}


def _clean_tokens(col):
    """Split on whitespace and drop empty + noise tokens, as a list column."""
    return (
        pl.col(col)
        .str.split(" ")
        .list.eval(
            pl.element()
            .filter(pl.element().str.len_chars() > 0)
            .filter(~pl.element().is_in(LEAD_NOISE))
        )
    )


def clean_first_token(df, alias="cf"):
    """First identifying token after noise removal, with a safe fallback.

    Fallback order (important: a name can be ALL noise, e.g. "Private Ltd"):
      1. first token after leading-noise removal
      2. whole normalized name (if that is empty too)
      3. the raw normalized first token
    """
    return df.with_columns(
        _clean_tokens("nn").list.head(1).list.join(" ").alias("_cf")
    ).with_columns(
        pl.when(pl.col("_cf").str.len_chars() > 0)
        .then(pl.col("_cf"))
        .otherwise(pl.col("nn"))
        .alias(alias)
    )


def norm_name(df):
    """Lowercase, strip punctuation, collapse whitespace."""
    return df.with_columns(
        pl.col("business_name")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("nn")
    )


def _first_tokens(col, n):
    """First n non-empty whitespace tokens of `col`, joined back together."""
    return (
        pl.col(col)
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.head(n)
        .list.join(" ")
    )


# ---------------------------------------------------------------------------
# Findings (diag_blocking.py + blocking_eda.py sweeps)
#
# 1. name_p4's 7,410 candidates/S1 is NOT caused by a few mega-buckets.
#    148,707 distinct keys, mean bucket 69.4 -- but the top 500 buckets hold
#    50% of all records and 65.9% of S1 entities sit in a bucket of >=1,000.
#    It is a broad, fat tail, not a handful of outliers.
#
# 2. The worst buckets are legal-form and honorific tokens used as the FIRST
#    word: us|the (91k), india|private (87k), us|llc (82k), india|m (55k,
#    from "M/s ..."), india|shri/sri/dr/mr/smt (~55k each).
#    So the user's stopword hypothesis was right.
#
# 3. BUT stripping them did NOT reduce candidate count:
#       name_p4  -> clean_p4 :  7,410 -> 7,744 candidates/S1
#    Recall rose 77.1% -> 83.2% (and F0.5 ceiling 0.898 -> 0.927), but volume
#    did not drop. Removing a token from the key just re-partitions the same
#    mass into the next-popular first token ("pediatric", "international",
#    "new", ...). The noise words were a symptom of low prefix entropy, not
#    the cause of candidate volume.
#
# 4. What DOES control volume is prefix LENGTH, because it sets entropy:
#       clean_p4  (4 chars) -> 7,744 cand, 83.2% recall
#       clean_p6  (6 chars) -> 5,953 cand, 81.6% recall
#       clean_full          ->    77 cand, 47.2% recall
#    clean_full / clean_sorted land near the <30 target but recall collapses,
#    because exact full-name equality is far too strict for this data.
#
# 5. Best measured trade-off so far:
#       UNION(clean_p4, clean_full, postal)
#         pair recall 95.09%, entity recall 86.27%, F0.5 ceiling 0.9811
#         exact dedup union: avg 19,244 / p95 44,762 / MAX 301,588 per S1
#    This clears the recall bar but is ~650x over the <30 candidate target.
#    The max of 301,588 is a real OOM risk even though the mean looks linear.
#
# CONCLUSION: the deterministic-equality blocking target as specified
# (>90% recall AND <30 candidates) is NOT achievable on this data with
# these key families. Equality keys trade recall against volume along one
# curve; no point on it satisfies both constraints. Breaking the curve
# requires either a within-bucket ranking stage (cheap first pass: build
# cheap features, keep top-K per S1 by score) or soft/vector blocking
# (phonetic + character n-grams via ANN). Recommend:
#   stage 1: clean_p4 + clean_full  -> generate pairs
#   stage 2: score cheaply, keep top ~30 per S1  -> feed the classifier
# ---------------------------------------------------------------------------


KEY_BUILDERS = {
    # ---- CLEANED variants (post-diagnosis) --------------------------------
    # 4 chars of the first *identifying* token, after noise stripping
    "clean_p4": lambda df: clean_first_token(norm_name(df)).with_columns(
        (
            pl.col("country").fill_null("").str.to_lowercase()
            + "|"
            + pl.col("cf").str.slice(0, 4)
        ).alias("k")
    ),
    # 6 chars -- more discriminative, still robust to light typos at the tail
    "clean_p6": lambda df: clean_first_token(norm_name(df)).with_columns(
        (
            pl.col("country").fill_null("").str.to_lowercase()
            + "|" + pl.col("cf").str.slice(0, 6)
        ).alias("k")
    ),
    # full cleaned name: country | all noise-free tokens, order preserved
    "clean_full": lambda df: norm_name(df).with_columns(
        _clean_tokens("nn").list.join(" ").alias("cf")
    ).with_columns(
        (
            pl.col("country").fill_null("").str.to_lowercase()
            + "|" + pl.col("cf")
        ).alias("k")
    ),
    # sorted cleaned tokens: catches word-order transpositions
    "clean_sorted": lambda df: norm_name(df).with_columns(
        _clean_tokens("nn").list.sort().list.join(" ").alias("cf")
    ).with_columns(
        (
            pl.col("country").fill_null("").str.to_lowercase()
            + "|" + pl.col("cf")
        ).alias("k")
    ),
    # ---- ORIGINAL baselines, kept for comparison --------------------------
    # country only -- deliberately loose baseline
    "country": lambda df: df.with_columns(
        pl.col("country").str.to_lowercase().alias("k")
    ),
    # 4 chars of the first name token
    "name_p4": lambda df: norm_name(df).with_columns(
        (
            pl.col("country").str.to_lowercase()
            + "|"
            + _first_tokens("nn", 1).str.slice(0, 4)
        ).alias("k")
    ),
    # first 2 tokens, 7 chars each
    "name_p3x2": lambda df: norm_name(df).with_columns(
        (
            pl.col("country").str.to_lowercase()
            + "|"
            + _first_tokens("nn", 2)
            .str.replace_all(" ", "_", literal=True)
            .str.slice(0, 7)
        ).alias("k")
    ),
    # last name token -- catches DBA / suffix reordering
    "name_last": lambda df: norm_name(df).with_columns(
        (
            pl.col("country").str.to_lowercase()
            + "|"
            + pl.col("nn").str.split(" ").list.last().str.slice(0, 4)
        ).alias("k")
    ),
    # postal-ish code: US ZIP / India PIN / France postal
    "postal": lambda df: df.with_columns(
        (
            pl.col("country").str.to_lowercase()
            + "|"
            + pl.col("business_address")
            .str.replace_all(r"\D", "")
            .str.replace_all(r"^0+", "")
            .str.slice(0, 6)
        ).alias("k")
    ),
    # name prefix + postal prefix (composite, usually higher recall)
    "name_p4_x_postal": lambda df: norm_name(df).with_columns(
        (
            pl.col("country").str.to_lowercase()
            + "|"
            + _first_tokens("nn", 1).str.slice(0, 4)
            + "|"
            + pl.col("business_address")
            .str.replace_all(r"\D", "")
            .str.replace_all(r"^0+", "")
            .str.slice(0, 3)
        ).alias("k")
    ),
}


def evaluate_key(name, builder, train_dir, verbose=True):
    t0 = time.time()
    scan = {
        s: pl.scan_csv(os.path.join(train_dir, f"train_source{s}.tsv"), separator="\t")
        for s in (1, 2, 3)
    }
    gt = pl.scan_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")

    s1k = builder(scan[1]).select(["entity_id", "k"]).collect()
    s2k = builder(scan[2]).select(["entity_id", "k"]).collect()
    s3k = builder(scan[3]).select(["entity_id", "k"]).collect()
    s23k = pl.concat([s2k, s3k], how="vertical")
    n_s1 = s1k.height
    if verbose:
        print(f"  loaded s1={n_s1:,} s2={s2k.height:,} s3={s3k.height:,}")

    # ---- (a) candidate load: rows per key, joined onto S1 (no cross product) --
    key_counts = s23k.group_by("k").agg(pl.len().alias("cnt"))
    per_s1 = s1k.join(key_counts, on="k", how="left").with_columns(
        pl.col("cnt").fill_null(0).cast(pl.UInt64)
    )
    cnt = per_s1["cnt"]
    avg_cand = cnt.mean() or 0.0
    p95_cand = cnt.quantile(0.95) or 0.0
    max_cand = int(cnt.max() or 0)
    # cast to u64 BEFORE summing: loose keys overflow u32 (4.2B) and wrap
    total_pairs = int(cnt.sum() or 0)
    zero_frac = float((cnt == 0).mean() or 0.0)

    # ---- (b) recall: exploded ground truth vs per-entity keys ----------------
    gtx = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("matched_entity_ids")])
        .collect()
    )
    total_true = gtx.height

    hit = (
        gtx.join(
            s1k.rename({"entity_id": "source1_entity_id", "k": "k1"}),
            on="source1_entity_id",
            how="left",
        )
        .join(
            s23k.rename({"entity_id": "matched_entity_ids", "k": "k2"}),
            on="matched_entity_ids",
            how="left",
        )
        .with_columns((pl.col("k1") == pl.col("k2")).alias("hit"))
    )
    pair_recall = float(hit["hit"].mean() or 0.0)

    per_ent = hit.group_by("source1_entity_id").agg(
        pl.len().alias("n_true"), pl.col("hit").sum().alias("n_hit")
    )
    entity_recall = float((per_ent["n_hit"] == per_ent["n_true"]).mean() or 0.0)

    # macro-F0.5 ceiling: PERFECT classifier on surviving candidates.
    # Singletons (no true match) always score 1.0.
    n_singletons = n_s1 - per_ent.height
    prec = 1.0
    rec_full = per_ent["n_hit"] / per_ent["n_true"]
    f_each = ((1.25 * prec * rec_full) / (0.25 * prec + rec_full)).fill_null(0.0)
    f_ceiling = float((f_each.sum() + n_singletons) / n_s1)

    res = {
        "key": name,
        "pair_recall": pair_recall,
        "entity_recall": entity_recall,
        "f05_ceiling": f_ceiling,
        "avg_cand": avg_cand,
        "p95_cand": p95_cand,
        "max_cand": max_cand,
        "total_pairs": total_pairs,
        "zero_frac": zero_frac,
        "secs": time.time() - t0,
    }
    if verbose:
        print(
            f"  pairRec={pair_recall*100:.2f}%  entRec={entity_recall*100:.2f}%"
            f"  F0.5ceil={f_ceiling:.4f}  avgCand={avg_cand:.2f}"
            f"  p95={p95_cand:.0f}  pairs={total_pairs:,}  ({res['secs']:.0f}s)"
        )
    del s1k, s2k, s3k, s23k, key_counts, per_s1, gtx, hit, per_ent
    gc.collect()
    return res


def union_evaluate(key_names, train_dir, verbose=True):
    """Evaluate the UNION of several keys (candidate set = OR of the parts).

    A true match is kept if ANY key matches on both sides. Recall is therefore
    the max over the parts, while candidate count is the sum minus overlap.
    """
    t0 = time.time()
    builders = [(n, KEY_BUILDERS[n]) for n in key_names]

    def multikey(df):
        """Apply every builder in sequence; returns a collected DataFrame.

        Each builder chains on a LazyFrame, but subsequent joins need eager
        frames, so collect at the end of the chain.
        """
        for i, (_, b) in enumerate(builders):
            df = b(df).rename({"k": f"k{i}"})
        return df.collect()

    scan = {
        s: pl.scan_csv(os.path.join(train_dir, f"train_source{s}.tsv"), separator="\t")
        for s in (1, 2, 3)
    }
    gt = pl.scan_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")

    kcols = [f"k{i}" for i in range(len(builders))]
    s1k = multikey(scan[1]).select(["entity_id", *kcols])
    s23k = pl.concat(
        [multikey(scan[2]).select(["entity_id", *kcols]),
         multikey(scan[3]).select(["entity_id", *kcols])],
        how="vertical",
    )
    n_s1 = s1k.height

    # candidate count: explode each S23 row across its keys, count, sum per S1
    s23_long = s23k.unpivot(
        index=["entity_id"], on=kcols, variable_name="part", value_name="k"
    ).filter(pl.col("k").is_not_null())
    key_counts = s23_long.group_by("k").agg(pl.len().alias("cnt"))
    s1_long = s1k.unpivot(
        index=["entity_id"], on=kcols, variable_name="part", value_name="k"
    ).filter(pl.col("k").is_not_null())

    per_s1 = s1_long.join(key_counts, on="k", how="left").with_columns(
        pl.col("cnt").fill_null(0).cast(pl.UInt64)
    )
    cnt = per_s1.group_by("entity_id").agg(pl.col("cnt").sum().alias("c"))["c"]
    avg_cand = float(cnt.mean() or 0.0)
    p95_cand = float(cnt.quantile(0.95) or 0.0)
    max_cand = int(cnt.max() or 0)
    total_pairs = int(cnt.sum() or 0)

    # recall: keep the pair if ANY part key agrees
    gtx = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("matched_entity_ids")])
        .collect()
    )
    a = s1k.rename({"entity_id": "source1_entity_id"})
    b = s23k.rename({"entity_id": "matched_entity_ids"})
    hit = gtx.join(a, on="source1_entity_id", how="left").join(
        b, on="matched_entity_ids", how="left"
    )
    hit = hit.with_columns(
        pl.any_horizontal([pl.col(c + "_right") == pl.col(c) for c in kcols]).alias("hit")
    )
    pair_recall = float(hit["hit"].mean() or 0.0)
    per_ent = hit.group_by("source1_entity_id").agg(
        pl.len().alias("n_true"), pl.col("hit").sum().alias("n_hit")
    )
    entity_recall = float((per_ent["n_hit"] == per_ent["n_true"]).mean() or 0.0)
    n_sing = n_s1 - per_ent.height
    rf = per_ent["n_hit"] / per_ent["n_true"]
    f_each = ((1.25 * 1.0 * rf) / (0.25 * 1.0 + rf)).fill_null(0.0)
    f_ceiling = float((f_each.sum() + n_sing) / n_s1)

    res = {
        "key": "UNION(" + "+".join(key_names) + ")",
        "pair_recall": pair_recall,
        "entity_recall": entity_recall,
        "f05_ceiling": f_ceiling,
        "avg_cand": avg_cand,
        "p95_cand": p95_cand,
        "max_cand": max_cand,
        "total_pairs": total_pairs,
        "zero_frac": 0.0,
        "secs": time.time() - t0,
    }
    if verbose:
        print(
            f"  pairRec={pair_recall*100:.2f}%  entRec={entity_recall*100:.2f}%"
            f"  F0.5ceil={f_ceiling:.4f}  avgCand={avg_cand:.2f}"
            f"  p95={p95_cand:.0f}  pairs={total_pairs:,}  ({res['secs']:.0f}s)"
        )
    del s1k, s23k, s23_long, s1_long, key_counts, gtx, hit, per_ent
    gc.collect()
    return res


def union_exact_sample(key_names, train_dir, n_sample=50_000, verbose=True):
    """EXACT deduplicated union size, measured on a random sample of S1.

    union_evaluate() SUMS the per-key candidate counts, so a candidate that
    shares two keys is counted twice. That is an upper bound, not the true
    union size. Here we materialise the real pair set for a sample of S1
    entities and de-duplicate, giving the true candidates-per-entity figure
    (the number that actually matters for runtime and for candidate_pairs.tsv,
    where duplicate IDs are a submission error).
    """
    t0 = time.time()
    builders = [(n, KEY_BUILDERS[n]) for n in key_names]
    kcols = [f"k{i}" for i in range(len(builders))]

    def multikey(df):
        for i, (_, b) in enumerate(builders):
            df = b(df).rename({"k": f"k{i}"})
        return df.collect()

    s1 = multikey(pl.scan_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t"))
    s23 = pl.concat(
        [multikey(pl.scan_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")),
         multikey(pl.scan_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t"))],
        how="vertical",
    )
    s1 = s1.filter(pl.col("country").is_not_null()).sample(n=min(n_sample, s1.height), seed=42)
    s1 = s1.with_row_index("_rid")
    sample_ids = set(s1["entity_id"].to_list())

    # build the real pair set: unpivot both sides onto a single key column
    a = s1.unpivot(index=["_rid", "entity_id"], on=kcols, value_name="k").filter(pl.col("k").is_not_null())
    b = s23.unpivot(index=["entity_id"], on=kcols, value_name="k").filter(pl.col("k").is_not_null())
    pairs = (
        a.join(b, on="k", how="inner")
        .select(["_rid", pl.col("entity_id_right").alias("cand")])
        .unique()
    )
    per = pairs.group_by("_rid").agg(pl.len().alias("c"))["c"]
    n = s1.height
    avg = float(per.mean() or 0.0)
    p95 = float(per.quantile(0.95) or 0.0)
    mx = int(per.max() or 0)
    tot = int(per.sum() or 0)
    zero = n - per.len()
    if verbose:
        print(
            f"  EXACT union over {n:,} sampled S1: avg={avg:.2f} p95={p95:.0f} "
            f"max={mx:,} total={tot:,} zero_cand={zero/n*100:.1f}% ({time.time()-t0:.0f}s)"
        )
    del a, b, pairs, per, s1, s23
    gc.collect()
    return {"n": n, "avg": avg, "p95": p95, "max": mx, "total": tot,
            "zero_frac": zero / n, "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--keys", default="all", help="comma list of key names, or 'all'")
    ap.add_argument(
        "--union",
        action="append",
        default=[],
        help="comma list of keys to UNION; repeatable. e.g. --union name_p4,postal",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=50_000,
        help="S1 rows to use for the exact dedup union measurement",
    )
    args = ap.parse_args()

    names = list(KEY_BUILDERS) if args.keys == "all" else args.keys.split(",")
    print(f"Polars {pl.__version__} | train-dir={args.train_dir}\n")
    print("=" * 112)
    print(
        f"{'key':<20}{'pairRec':>9}{'entRec':>9}{'F0.5ceil':>10}"
        f"{'avgCand':>10}{'p95':>8}{'maxCand':>9}{'totalPairs':>16}{'secs':>7}"
    )
    print("-" * 112)

    results = []
    for n in names:
        if n not in KEY_BUILDERS:
            print(f"!! unknown key {n!r}; known: {', '.join(KEY_BUILDERS)}")
            continue
        print(f"[{n}]")
        try:
            results.append(evaluate_key(n, KEY_BUILDERS[n], args.train_dir))
        except Exception as exc:  # a bad key shouldn't kill the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("-" * 112)

    if args.union:
        for spec in args.union:
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            bad = [p for p in parts if p not in KEY_BUILDERS]
            if bad:
                print(f"!! union: unknown key(s) {bad}; skipping")
                continue
            print(f"[UNION {'+'.join(parts)}]")
            try:
                results.append(union_evaluate(parts, args.train_dir))
                union_exact_sample(parts, args.train_dir, n_sample=args.sample)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
            print("-" * 112)

    if results:
        results.sort(key=lambda r: -r["f05_ceiling"])
        print("\nRANKED by macro-F0.5 ceiling (the metric that actually matters):")
        for r in results:
            print(
                f"  {r['key']:<20} ceiling={r['f05_ceiling']:.4f}  "
                f"pairRec={r['pair_recall']*100:5.2f}%  avgCand={r['avg_cand']:9.2f}  "
                f"pairs={r['total_pairs']:>14,}"
            )


if __name__ == "__main__":
    main()
