"""
Stage 4: chunked TEST inference -> output/matching_results.tsv + candidate_pairs.tsv

Architecture (as agreed):
  STEP 0  precompute the S2/S3 key index ONCE -> test_s23_keys.parquet
  STEP 1  iterate test S1 in chunks, lazy-join against that cached index
  STEP 2  hard-cap candidates per S1 (.head) BEFORE featurising
  STEP 3  featurise (identical features to training), predict, threshold
  STEP 4  append to the two output TSVs

FINAL MODEL: clf_model_v2.txt, threshold 0.64, cap 1000.
  v1 (0.9319 val F0.5) was trained with only 40 negatives/S1 but INFERENCE
  feeds up to 1000 candidates, so it over-predicted badly (89 matches/S1 vs
  3.7 expected). v2 is trained with 200 hard + 200 random negatives per S1
  (202 rows/S1), matching inference scale. Measured on a 5k test slice:
  4.66 matches/S1, 6.2% singletons -- consistent with the train distribution
  (3.67 matches, 5.6% singletons). Validation macro F0.5 = 0.8517.

WHY THE BUCKET GUARD (--max-bucket)
  k2 = country|postal-prefix has buckets of 10k-100k+ members. Joining it for
  every chunk was ~40 hours on its own and OOMed at 25k-row chunks. We
  precompute per-key bucket sizes ONCE and skip any key over the threshold;
  k0/k1 still cover those entities. This cut the run to ~14.5h.

PERFORMANCE NOTES (15GB RAM, 16 cores)
  - index build: ~7s for 9.97M rows -> 719MB parquet
  - throughput: ~33 S1 rows/sec at chunk=2500
  - use semi-joins (hash), not is_in (full scan), to restrict the right side

OUTPUT CONTRACT (enforced by utils/validate_submission.py):
  * tab separated, UTF-8, header exactly:
      source1_entity_id\tmatched_entity_ids
      source1_entity_id\tcandidate_entity_ids
  * EVERY S1 entity gets exactly one row, even if it has no candidates
  * empty second column for singletons (trailing tab, then newline)
  * candidate_pairs.tsv = the pool fed to the model (post hard-cap,
    PRE threshold) so every matched id is a subset of the candidates
  * no trailing/duplicate commas, no duplicate ids within a list
"""

import argparse
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

TEST = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\test"
OUT = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\output"

# must match clf_pipeline.FEATURES exactly
FEATURES = [
    "name_token_sort", "name_jaro_winkler", "name_lev_ratio", "name_len_diff",
    "name_jac3", "addr_token_set", "addr_jac3", "addr_num_overlap",
    "addr_s23_missing", "country_same", "s",
]

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
    """Identical normalisation to clf_pipeline.norm (features depend on it)."""
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
        .with_columns(pl.col("country").fill_null("").str.to_lowercase().alias("ctry"))
        .with_columns(pl.col("business_address").fill_null("").alias("addr"))
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


def build_s23_index(index_path, force=False):
    """STEP 0: one pass over test S2/S3 -> cached parquet key index."""
    if os.path.exists(index_path) and not force:
        print(f"[index] reusing {index_path}")
        return
    t0 = time.time()
    s23 = add_keys(norm(pl.concat([
        pl.scan_csv(os.path.join(TEST, "test_source2.tsv"), separator="\t"),
        pl.scan_csv(os.path.join(TEST, "test_source3.tsv"), separator="\t"),
    ], how="vertical"))).collect()
    print(f"[index] S2/S3 rows: {s23.height:,} ({time.time()-t0:.0f}s)", flush=True)
    s23.select(["entity_id", "business_name", "addr", "nn", "ctry", "ft",
                "nchar", "ntok", *KEYS]).write_parquet(index_path)
    print(f"[index] wrote {index_path} "
          f"({os.path.getsize(index_path)/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)
    del s23
    gc.collect()


def add_features(df):
    """Must mirror clf_pipeline.add_features exactly."""
    n1 = df["business_name_x"].to_list()
    n2 = df["business_name_y"].to_list()
    a1 = [x or "" for x in df["addr_x"].to_list()]
    a2 = [x or "" for x in df["addr_y"].to_list()]
    nn1 = df["nn_x"].to_list()
    nn2 = df["nn_y"].to_list()

    out = df.with_columns(
        pl.Series("name_token_sort", [fuzz.token_sort_ratio(a, b) for a, b in zip(nn1, nn2)], dtype=pl.Float32),
        pl.Series("name_jaro_winkler", [JaroWinkler.normalized_similarity(a, b) for a, b in zip(n1, n2)], dtype=pl.Float32),
        pl.Series("name_lev_ratio", [fuzz.ratio(a, b) for a, b in zip(nn1, nn2)], dtype=pl.Float32),
        pl.Series("addr_token_set", [fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)], dtype=pl.Float32),
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
        pl.Series("name_jac3", [jac3(a, b) for a, b in zip(nn1, nn2)], dtype=pl.Float32),
        pl.Series("addr_jac3", [jac3(a, b) for a, b in zip(a1, a2)], dtype=pl.Float32),
        pl.Series("addr_num_overlap", [numov(a, b) for a, b in zip(a1, a2)], dtype=pl.Float32),
    )

KEYS = ["k0", "k1", "k2"]



def join_id_list(ids):
    """Format an id list: comma separated, no trailing comma, no dupes."""
    if not ids:
        return ""
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return ",".join(out)


def build_bucket_sizes(s23, max_bucket):
    """Precompute per-key bucket sizes ONCE so each chunk can skip huge keys."""
    out = {}
    for k in KEYS:
        c = s23.group_by(k).agg(pl.len().alias("n"))
        big = c.filter(pl.col("n") > max_bucket)
        d = {r[0]: r[1] for r in big.iter_rows()}
        out[k] = d
        print(f"[buckets] {k}: {len(d):,} keys exceed {max_bucket:,} members",
              flush=True)
    return out


def process_chunk(part, s23, clf, cap, thr, bucket_sizes=None, max_bucket=100_000):
    """Returns (s1_ids, cand_map, match_map) for one S1 chunk.

    cand_map : s1_id -> candidate id list (post hard-cap, PRE threshold)
    match_map: s1_id -> matched id list (prob >= thr)
    """
    # Do the three keys as SEPARATE joins, capping per key. An unpivot join
    # multiplies rows badly, and k2 (country|postal-prefix) alone emitted
    # ~10M pairs per 25k chunk -> OOM at 15GB. Restrict the candidate side to
    # this chunk's keys BEFORE joining so the cross product never materialises.
    sel_base = ["entity_id", "ft", "nchar", "ntok"]
    per_key = []
    for k in KEYS:
        aa = part.select([*sel_base, k]).rename({"entity_id": "s1"})
        aa = aa.filter(pl.col(k).is_not_null())
        if aa.height == 0:
            continue
        # SKIP PATHOLOGICAL BUCKETS. Precomputed sizes (bucket_sizes) let us
        # drop a key whose bucket is huge BEFORE joining: the country|postal
        # key k2 routinely has 10k+ members, and joining it for every chunk
        # was ~40 hours of work on its own. k0/k1 still cover those entities.
        if bucket_sizes is not None:
            sizes = bucket_sizes.get(k) or {}
            if sizes:
                # sizes maps key -> size for keys ALREADY over max_bucket
                bad = [x for x in aa[k].unique().to_list() if x in sizes]
                if bad:
                    aa = aa.filter(~pl.col(k).is_in(pl.Series(bad).implode()))
            if aa.height == 0:
                continue
        # semi-join (hash) instead of is_in (scan) to restrict the right side
        used = aa.select(k).unique().rename({k: "k"})
        bb = s23.join(used, left_on=k, right_on="k", how="semi") \
                .select([*sel_base, k]).rename({"entity_id": "cand"})
        j = (aa.join(bb, on=k, how="inner", suffix="_r")
               .select([
                   pl.col("s1"), pl.col("cand"),
                   pl.col("ft").alias("ft1"), pl.col("ft_r").alias("ft2"),
                   pl.col("nchar").alias("nc1"), pl.col("nchar_r").alias("nc2"),
                   pl.col("ntok").alias("nt1"), pl.col("ntok_r").alias("nt2"),
               ])
               .unique(subset=["s1", "cand"]))
        if j.height:
            per_key.append(
                j.sort("s1").group_by("s1", maintain_order=True).head(cap))
        del aa, bb, j
        gc.collect()
    if not per_key:
        return part["entity_id"].to_list(), {}, {}
    pairs = pl.concat(per_key, how="vertical").unique(subset=["s1", "cand"])
    del per_key
    gc.collect()
    if pairs.height == 0:
        return part["entity_id"].to_list(), {}, {}

    pairs = pairs.with_columns(
        (pl.col("ft1") == pl.col("ft2")).cast(pl.UInt8).alias("f_eq"),
        (pl.col("nt1") == pl.col("nt2")).cast(pl.UInt8).alias("t_eq"),
        (pl.min_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32)
         / pl.max_horizontal(pl.col("nc1"), pl.col("nc2")).cast(pl.Float32).clip(1.0)
         ).alias("len_r"),
    ).with_columns(
        (3.0 * pl.col("f_eq") + 1.0 * pl.col("t_eq") + 1.5 * pl.col("len_r")).alias("s")
    )
    # HARD CAP before featurising -- bounds memory and fixes the output size
    pairs = (pairs.sort("s1", "s", descending=[False, True])
                  .group_by("s1", maintain_order=True).head(cap))
    if pairs.height == 0:
        return part["entity_id"].to_list(), {}, {}

    # candidate_pairs.tsv content = exactly this pre-threshold pool
    # NOTE: iterating a Polars group_by yields TUPLE keys, e.g.
    # ('S1-123',) not 'S1-123'. Unwrap or every downstream dict lookup misses
    # and the output silently comes out empty.
    cand_map = {}
    for key, grp in pairs.group_by("s1", maintain_order=True):
        s1id = key[0] if isinstance(key, tuple) else key
        cand_map[s1id] = grp["cand"].to_list()

    feat = add_features(
        pairs.join(part.select(["entity_id", "business_name", "addr", "nn", "ctry"])
                   .rename({"entity_id": "s1", "business_name": "business_name_x",
                            "addr": "addr_x", "nn": "nn_x", "ctry": "ctry_x"}),
                   on="s1", how="left")
             .join(s23.select(["entity_id", "business_name", "addr", "nn", "ctry"])
                   .rename({"entity_id": "cand", "business_name": "business_name_y",
                            "addr": "addr_y", "nn": "nn_y", "ctry": "ctry_y"}),
                   on="cand", how="left")
    )
    X = feat.select(FEATURES).to_numpy()
    # LGBMBooster has no predict_proba; positive-class prob comes from
    # clf.predict(...) which returns P(y=1) for a binary objective.
    prob = clf.predict(X)
    feat = feat.with_columns(pl.Series("prob", prob))

    match_map = {}
    keep = feat.filter(pl.col("prob") >= thr)
    for key, grp in keep.group_by("s1", maintain_order=True):
        s1id = key[0] if isinstance(key, tuple) else key
        match_map[s1id] = grp["cand"].to_list()

    return part["entity_id"].to_list(), cand_map, match_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=TEST)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--index", default="test_s23_keys.parquet")
    ap.add_argument("--model", default="clf_model.txt")
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--thr", type=float, default=0.66)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N s1 rows")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--max-bucket", type=int, default=100_000,
                    help="skip a blocking key whose bucket is larger than this")
    args = ap.parse_args()
    t0 = time.time()

    os.makedirs(args.out_dir, exist_ok=True)
    build_s23_index(args.index, force=args.rebuild_index)

    clf = lgb.Booster(model_file=args.model)
    clf.predict(np.zeros((1, len(FEATURES)), dtype=np.float32))  # warm up

    s23 = pl.read_parquet(args.index)
    print(f"[infer] index rows: {s23.height:,}", flush=True)
    bsz = build_bucket_sizes(s23, args.max_bucket)

    s1lf = pl.scan_csv(os.path.join(args.test_dir, "test_source1.tsv"), separator="\t")
    if args.limit:
        s1lf = s1lf.head(args.limit)
    total = s1lf.select(pl.len()).collect().item()
    print(f"[infer] test S1 rows: {total:,}", flush=True)

    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    n_written = n_sing = n_pairs = 0

    with open(mpath, "w", encoding="utf-8", newline="\n") as mf, \
         open(cpath, "w", encoding="utf-8", newline="\n") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

        n_chunks = (total + args.chunk - 1) // args.chunk
        for ci, part in enumerate(s1lf.collect().iter_slices(args.chunk)):
            part = add_keys(norm(part))
            s1ids, cand_map, match_map = process_chunk(
                part, s23, clf, args.cap, args.thr,
                bucket_sizes=bsz, max_bucket=args.max_bucket)
            for s1id in s1ids:
                # ONE row per S1 entity, always. Empty string => singleton.
                mf.write(f"{s1id}\t{join_id_list(match_map.get(s1id, []))}\n")
                cf.write(f"{s1id}\t{join_id_list(cand_map.get(s1id, []))}\n")
                n_written += 1
                if not match_map.get(s1id):
                    n_sing += 1
                n_pairs += len(match_map.get(s1id, []))
            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                el = time.time() - t0
                rate = n_written / max(el, 1e-9)
                eta = (total - n_written) / max(rate, 1e-9)
                print(f"  chunk {ci+1}/{n_chunks} rows={n_written:,} "
                      f"singletons={n_sing:,} ({n_sing/max(n_written,1)*100:.1f}%) "
                      f"matches={n_pairs:,} {rate:.0f}/s eta={eta/60:.0f}min", flush=True)
            del part, cand_map, match_map
            gc.collect()

    print(f"\n[infer] wrote {n_written:,} rows in {time.time()-t0:.0f}s")
    print(f"[infer] singletons: {n_sing:,} ({n_sing/max(n_written,1)*100:.2f}%)")
    print(f"[infer] total matched pairs: {n_pairs:,} "
          f"({n_pairs/max(n_written,1):.2f} per S1)")
    print(f"[infer] {mpath}\n[infer] {cpath}")


if __name__ == "__main__":
    main()

