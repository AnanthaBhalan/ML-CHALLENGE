"""
Stage 1: candidate pool + labelling for the LightGBM pair classifier.

Builds, for a SAMPLE of S1 entities, the union-of-keys candidate pool (same
keys that measured 95.09% pair recall / 0.9811 ceiling in blocking_eda.py),
labels every pair against train_ground_truth.tsv, and samples negatives.

Memory strategy: candidates are generated in CHUNKS of S1 entities and
immediately subsampled (all true positives + hard negatives + random
negatives), so the ~19k candidates/entity never all coexist in memory.

Writes a parquet cache so the feature/training stage can re-run cheaply.
"""

import argparse
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
KEYS = ["k0", "k1", "k2"]


def norm(df):
    return (
        df.with_columns(
            pl.col("business_name").fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^\w\s]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars().alias("nn")
        )
        .with_columns(pl.col("nn").str.split(" ").alias("_all"))
        .with_columns(
            pl.col("_all").list.eval(
                pl.element().filter(pl.element().str.len_chars() > 0)
                .filter(~pl.element().is_in(LEAD_NOISE))
            ).alias("toks")
        )
        .with_columns(pl.col("toks").list.head(1).list.join(" ").alias("_ft"))
        .with_columns(
            pl.when(pl.col("_ft").str.len_chars() > 0)
            .then(pl.col("_ft")).otherwise(pl.col("nn")).alias("ft")
        )
        .with_columns(
            pl.col("country").fill_null("").str.to_lowercase().alias("ctry")
        )
        .with_columns(
            pl.col("business_address").fill_null("").alias("addr")
        )
        .with_columns(
            pl.col("nn").str.len_chars().cast(pl.UInt16).alias("nchar"),
            pl.col("toks").list.len().cast(pl.UInt16).alias("ntok"),
        )
    )


def add_keys(df):
    return df.with_columns(
        (pl.col("ctry") + "|" + pl.col("ft").str.slice(0, 4)).alias("k0"),
        (pl.col("ctry") + "|" + pl.col("toks").list.join(" ")).alias("k1"),
        (pl.col("ctry") + "|" + pl.col("addr")
         .str.replace_all(r"\D", "").str.replace_all(r"^0+", "").str.slice(0, 3)).alias("k2"),
    )



def build_pairs_for_chunk(s1_chunk, s23, cap_pool=4000):
    """Union of the 3 keys -> de-duplicated (s1, cand) pairs for this chunk.

    A hard per-entity cap keeps the chunk bounded. Because it is a cap (not a
    ranking), true positives CAN be dropped -- so we report pool recall and
    keep the cap generous relative to the measured ~19k/S1.
    """
    idx1 = ["_rid", "ft", "nchar", "ntok", *KEYS]
    idx2 = ["_cid", "ft", "nchar", "ntok", *KEYS]
    a = s1_chunk.select(idx1).unpivot(index=idx1, on=KEYS, value_name="k") \
                .filter(pl.col("k").is_not_null())
    b = s23.select(idx2).unpivot(index=idx2, on=KEYS, value_name="k") \
            .filter(pl.col("k").is_not_null())
    sel = [
        pl.col("_rid"), pl.col("_cid"),
        pl.col("ft").alias("ft1"), pl.col("ft_c").alias("ft2"),
        pl.col("nchar").alias("nc1"), pl.col("nchar_c").alias("nc2"),
        pl.col("ntok").alias("nt1"), pl.col("ntok_c").alias("nt2"),
    ]
    pairs = a.join(b, on="k", how="inner", suffix="_c").select(sel) \
             .unique(subset=["_rid", "_cid"])
    pairs = pairs.with_columns(
        (pl.col("ft1") == pl.col("ft2")).cast(pl.UInt8).alias("f_eq"),
        (pl.col("nt1") == pl.col("nt2")).cast(pl.UInt8).alias("t_eq"),
        (pl.min_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32)
         / pl.max_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32).clip(1.0)
         ).alias("len_r"),
    ).with_columns(
        (3.0 * pl.col("f_eq") + 1.0 * pl.col("t_eq") + 1.5 * pl.col("len_r")).alias("s")
    )
    if cap_pool:
        pairs = (pairs.sort("_rid", "s", descending=[False, True])
                      .group_by("_rid", maintain_order=True).head(cap_pool))
    return pairs.select(["_rid", "_cid", "s"])


def sample_negatives(pairs, truth, n_hard=20, n_rand=20, seed=7):
    """Keep all true positives; sample hard + random negatives per S1 entity."""
    p = pairs.join(
        truth.select(["_rid", "_cid"]).with_columns(pl.lit(True).alias("y")),
        on=["_rid", "_cid"], how="left",
    ).with_columns(pl.col("y").fill_null(False))
    pos = p.filter(pl.col("y"))
    neg = p.filter(~pl.col("y"))

    # HARD negatives: highest cheap score among non-matches
    hard = (neg.sort("_rid", "s", descending=[False, True])
               .group_by("_rid", maintain_order=True).head(n_hard))
    # RANDOM negatives for calibration / false-positive modelling
    rnd = (neg.with_columns((pl.int_range(0, pl.len()) * 2654435761
                             + pl.col("_rid")) .alias("rnd"))
               .sort(["_rid", "rnd"])
               .group_by("_rid", maintain_order=True).head(n_rand)
               .drop("rnd"))
    out = pl.concat([pos, hard, rnd], how="vertical").unique(subset=["_rid", "_cid"])
    return out



# ---------------------------------------------------------------------------
# Features. rapidfuzz exposes C++ scorers ~100x faster than difflib, so we use
# it for the fuzzy string metrics and keep everything else in Polars/numpy.
# ---------------------------------------------------------------------------
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.distance import JaroWinkler  # noqa: E402 (moved out of fuzz in 3.x)
import numpy as np  # noqa: E402


def _pairwise(fn, a, b):
    return [fn(x, y) for x, y in zip(a, b)]


def add_features(df):
    """df must contain: business_name_x/_y, addr_x/_y, ctry_x/_y, nn_x/_y."""
    n1 = df["business_name_x"].to_list()
    n2 = df["business_name_y"].to_list()
    a1 = [x or "" for x in df["addr_x"].to_list()]
    a2 = [x or "" for x in df["addr_y"].to_list()]
    nn1 = df["nn_x"].to_list()
    nn2 = df["nn_y"].to_list()

    out = df.with_columns(
        pl.Series("name_token_sort", _pairwise(fuzz.token_sort_ratio, nn1, nn2), dtype=pl.Float32),
        pl.Series("name_jaro_winkler", _pairwise(JaroWinkler.normalized_similarity, n1, n2), dtype=pl.Float32),
        pl.Series("name_lev_ratio", _pairwise(fuzz.ratio, nn1, nn2), dtype=pl.Float32),
        pl.Series("addr_token_set", _pairwise(fuzz.token_set_ratio, a1, a2), dtype=pl.Float32),
        pl.Series("addr_s23_missing", [1 if y == "" else 0 for y in a2], dtype=pl.UInt8),
        pl.Series("country_same", [1 if x == y else 0 for x, y in zip(df["ctry_x"], df["ctry_y"])], dtype=pl.UInt8),
    )

    L1 = np.array([len(x) for x in nn1], dtype=np.float32)
    L2 = np.array([len(x) for x in nn2], dtype=np.float32)

    def jac3(x, y):
        if not x and not y:
            return 1.0
        gx = {x[i:i + 3] for i in range(max(len(x) - 2, 1))}
        gy = {y[i:i + 3] for i in range(max(len(y) - 2, 1))}
        u = len(gx | gy)
        return len(gx & gy) / u if u else 0.0

    def numov(x, y):
        sx = {t for t in "".join(c if c.isdigit() else " " for c in x).split() if len(t) >= 2}
        sy = {t for t in "".join(c if c.isdigit() else " " for c in y).split() if len(t) >= 2}
        if not sx and not sy:
            return 0.0
        return len(sx & sy) / len(sx | sy)

    return out.with_columns(
        pl.Series("name_len_diff", np.abs(L1 - L2) / np.maximum(1.0, np.maximum(L1, L2)), dtype=pl.Float32),
        pl.Series("name_jac3", [jac3(x, y) for x, y in zip(nn1, nn2)], dtype=pl.Float32),
        pl.Series("addr_jac3", [jac3(x, y) for x, y in zip(a1, a2)], dtype=pl.Float32),
        pl.Series("addr_num_overlap", [numov(x, y) for x, y in zip(a1, a2)], dtype=pl.Float32),
    )



import collections
import numpy as np  # noqa: E402


def macro_f05(y_true, pred_sets, n_entities):
    """Exact leaderboard metric: per-S1-entity F0.5, averaged over ALL entities.

    Singletons (no true matches) score 1.0 when predicted empty, 0.0 otherwise.
    """
    import collections
    true_by = collections.defaultdict(set)
    for e, c in y_true:
        true_by[e].add(c)
    pred_by = collections.defaultdict(set)
    for e, c in pred_sets:
        pred_by[e].add(c)
    scores = []
    for e in range(n_entities):
        t = true_by.get(e, set())
        p = pred_by.get(e, set())
        if not t and not p:
            scores.append(1.0)
            continue
        tp = len(t & p); fp = len(p - t); fn = len(t - p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if (prec == 0 and rec == 0)
                      else (1.25 * prec * rec) / (0.25 * prec + rec))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--cap-pool", type=int, default=4000)
    ap.add_argument("--n-hard", type=int, default=200)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--cache", default="clf_pool.parquet")
    args = ap.parse_args()
    t0 = time.time()

    s1 = add_keys(norm(pl.scan_csv(os.path.join(args.train_dir, "train_source1.tsv"),
                                  separator="\t"))).collect()
    s23 = add_keys(norm(pl.concat([
        pl.scan_csv(os.path.join(args.train_dir, "train_source2.tsv"), separator="\t"),
        pl.scan_csv(os.path.join(args.train_dir, "train_source3.tsv"), separator="\t"),
    ], how="vertical"))).collect().with_row_index("_cid")
    print(f"loaded s1={s1.height:,} s23={s23.height:,} ({time.time()-t0:.0f}s)", flush=True)

    s1 = s1.with_row_index("_rid").sample(fraction=1.0, shuffle=True, seed=42)
    s1 = s1.head(args.n_train + args.n_val)
    split = ["val"] * args.n_val + ["train"] * args.n_train
    s1 = s1.with_columns(pl.Series("_split", split[:s1.height]))
    print(f"S1 sampled: train~{args.n_train:,} val~{args.n_val:,}", flush=True)

    gt = (
        pl.scan_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), separator="\t")
        .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")])
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("mid")]).collect()
        .join(s1.select(["entity_id", "_rid"]), left_on="source1_entity_id",
              right_on="entity_id", how="inner")
        .select(["_rid", "mid"])
        .join(s23.select(["entity_id", "_cid"]), left_on="mid", right_on="entity_id", how="left")
        .drop_nulls("_cid")
    )
    print(f"true pairs in sample: {gt.height:,}", flush=True)

    parts = []
    n_chunks = (s1.height + args.chunk - 1) // args.chunk
    running = 0
    for ci, part in enumerate(s1.iter_slices(args.chunk)):
        pairs = build_pairs_for_chunk(part, s23, cap_pool=args.cap_pool)
        rids = part["_rid"].to_list()
        sub = sample_negatives(pairs, gt.filter(pl.col("_rid").is_in(rids)),
                               n_hard=args.n_hard, n_rand=args.n_rand)
        parts.append(sub)
        running += sub.height
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci+1}/{n_chunks} rows={running:,} ({time.time()-t0:.0f}s)",
                  flush=True)
    pool = pl.concat(parts, how="vertical")
    print(f"pool rows: {pool.height:,}  ({time.time()-t0:.0f}s)", flush=True)
    n_u = pool["_rid"].n_unique()
    print(f"rows per S1: {pool.height/max(n_u,1):.1f}  "
          f"(should roughly match the inference cap, else the model "
          f"over-predicts on the larger inference pool)", flush=True)

    pool = (pool.join(s1.select(["_rid", "_split", "business_name", "addr", "nn", "ctry"])
                      .rename({"business_name": "business_name_x", "addr": "addr_x",
                               "nn": "nn_x", "ctry": "ctry_x"}), on="_rid", how="left")
                .join(s23.select(["_cid", "business_name", "addr", "nn", "ctry"])
                      .rename({"business_name": "business_name_y", "addr": "addr_y",
                               "nn": "nn_y", "ctry": "ctry_y"}), on="_cid", how="left"))
    print("featurising...", flush=True)
    feat = add_features(pool)
    feat.write_parquet(args.cache)
    print(f"cached -> {args.cache}  rows={feat.height:,}  pos={int(feat['y'].sum()):,}  "
          f"({time.time()-t0:.0f}s)")


FEATURES = [
    "name_token_sort", "name_jaro_winkler", "name_lev_ratio", "name_len_diff",
    "name_jac3", "addr_token_set", "addr_jac3", "addr_num_overlap",
    "addr_s23_missing", "country_same", "s",
]


def train_and_sweep(cache="clf_pool.parquet", seed=7):
    import lightgbm as lgb
    import numpy as np

    d = pl.read_parquet(cache)
    tr = d.filter(pl.col("_split") == "train")
    va = d.filter(pl.col("_split") == "val")
    print(f"train rows={tr.height:,} pos={int(tr['y'].sum()):,} | "
          f"val rows={va.height:,} pos={int(va['y'].sum()):,}", flush=True)

    Xtr = tr.select(FEATURES).to_numpy()
    ytr = tr["y"].to_numpy().astype(np.int32)
    Xva = va.select(FEATURES).to_numpy()
    yva = va["y"].to_numpy().astype(np.int32)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "learning_rate": 0.05,
        "num_leaves": 63, "feature_fraction": 0.8,
        "n_estimators": 600, "n_jobs": -1, "verbosity": -1,
        "random_state": seed,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xva)[:, 1]
    print("trained.", flush=True)

    va = va.with_columns(pl.Series("prob", prob))
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1])
    print("\nfeature importance:")
    for f, i in imp:
        print(f"   {f:<20} {i}")

    # ---- threshold sweep on TRUE macro F0.5 --------------------------------
    n_val_entities = va["_rid"].n_unique()
    true_pairs = list(zip(va.filter(pl.col("y") == 1)["_rid"].to_list(),
                          va.filter(pl.col("y") == 1)["_cid"].to_list()))
    # remap _rid to a dense 0..N-1 space so every val entity is counted
    ids = va["_rid"].unique().sort().to_list()
    idmap = {r: i for i, r in enumerate(ids)}
    rr = [idmap[r] for r in va["_rid"].to_list()]
    true_pairs = [(idmap[a], b) for a, b in true_pairs]
    rr = np.array(rr)
    cc = va["_cid"].to_numpy()

    best = (0.0, 0.5)
    print(f"\nvalidating on {n_val_entities:,} S1 entities "
          f"(positives present: {len(true_pairs):,})", flush=True)
    for th in np.arange(0.30, 0.96, 0.02):
        m = prob >= th
        pred = list(zip(rr[m].tolist(), cc[m].tolist()))
        s = macro_f05(true_pairs, pred, len(ids))
        flag = ""
        if s > best[0]:
            best = (s, float(th)); flag = "  <-- best"
        print(f"  thr={th:.2f}  macroF0.5={s:.4f}{flag}", flush=True)
    print(f"\nOPTIMAL threshold={best[1]:.2f}  validation macro F0.5={best[0]:.4f}")

    # precision/recall at the optimum
    m = prob >= best[1]
    tp = int(((yva == 1) & m).sum()); fp = int(((yva == 0) & m).sum())
    fn = int(((yva == 1) & ~m).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    print(f"pair-level @opt: precision={p:.4f} recall={r:.4f} "
          f"(tp={tp:,} fp={fp:,} fn={fn:,})")
    clf.booster_.save_model("clf_model.txt")
    return best


if __name__ == "__main__":
    if os.environ.get("CLF_TRAIN_ONLY"):
        train_and_sweep(os.environ.get("CLF_CACHE", "clf_pool.parquet"))
    else:
        main()

