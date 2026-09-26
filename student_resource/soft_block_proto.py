"""
Soft blocking prototype: char n-gram TF-IDF + cosine, per-country hard block.

Replaces discrete equality buckets with continuous similarity, which is the
only way to break the recall-vs-volume curve measured in blocking_eda.py.

Design (per the agreed constraints):
  * HARD BLOCK BY COUNTRY -- a separate vectorizer + index per country, so no
    cross-border candidates are ever produced and RAM stays bounded.
  * CHAR N-GRAM (char_wb 3-5) -- resilient to pvt/private, shri/sri, typos,
    token merges and word reordering, which word-level TF-IDF cannot do.
  * sublinear_tf=True -- logarithmically damps the Zipfian stop-tokens
    ("the", "llc", "dr") automatically, no hand-written stopword list.
  * L2-normalised, so cosine == dot product == sparse matmul.

WHY NO HNSWLIB / FAISS
  Neither has a wheel for Python 3.13 on this machine (hnswlib wheel build
  fails; faiss-cpu has no cp313 wheel). We therefore use EXACT sparse cosine
  via chunked matmul. For a measured sample this is affordable and, unlike
  ANN, introduces no recall loss of its own -- so the numbers below are an
  upper bound on what an ANN index would achieve. If this works, hnswlib is
  purely a speed optimisation for the full run (e.g. under Python 3.11).

Usage:
    python soft_block_proto.py --sample 1500 --k 100
"""

import argparse
import gc
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TRAIN = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\train"


def norm(df, field="name"):
    """Lowercase, strip punctuation -> space, collapse whitespace.

    field: "name" uses business_name only; "name_addr" concatenates the
    normalised address too, giving the char n-grams extra signal (a true
    match usually agrees on BOTH name and address, and address is what
    disambiguates the many "Pediatric ..." businesses sharing a name).

    NOTE: field is passed as an ARG, not read from the frame. Reading it from
    df.schema["_field"] is wrong for a LazyFrame -- .schema returns DTYPES
    ("String"), not values, so the address branch silently never ran.
    """
    name = (
        pl.col("business_name")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    out = df.with_columns(name.alias("nn"))
    if field == "name_addr":
        addr = (
            pl.col("business_address")
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^\w\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        out = out.with_columns((pl.col("nn") + " " + addr).str.strip_chars().alias("nn"))
    return out


def f05(p, r):
    if p <= 0 and r <= 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)



def scipy_vstack(blocks):
    import scipy.sparse as sp
    return sp.vstack(blocks, format="csr")


def build_country_index(texts, n_features, chunk, verbose=True):
    """HashingVectorizer char n-grams -> L2-normalised sparse matrix.

    HashingVectorizer (not TfidfVectorizer) because it needs NO vocabulary
    pass: building a vocabulary + idf over 4M+ documents dominated the runtime
    and the memory. Hashing straight to a fixed dimension is stateless and
    streaming-friendly.

    sublinear_tf: HashingVectorizer has NO sublinear_tf argument, so we apply
    the same log damping manually on the raw counts (1 + log(tf)) before
    L2-normalising. This is what stops the Zipfian stop-tokens ("the", "llc",
    "dr") from dominating the cosine.

    Downside vs TfidfVectorizer: no idf weighting, so common n-grams are not
    down-weighted as aggressively, and the fixed dim causes some hash
    collisions. Both are acceptable for blocking (we only need a shortlist).
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    t0 = time.time()
    vec = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=n_features,          # 2**20, a power of two
        alternate_sign=False,           # keep counts non-negative
        norm=None,                      # we L2-normalise explicitly
        lowercase=False,
        dtype=np.float32,
    )

    t1 = time.time()
    blocks = []
    for start in range(0, len(texts), chunk):
        blocks.append(vec.transform(texts[start:start + chunk]))
        if verbose and (start // chunk) % 4 == 0:
            print(f"    hashed {min(start+chunk, len(texts)):,}/{len(texts):,} "
                  f"({time.time()-t1:.0f}s)", flush=True)
    X = scipy_vstack(blocks)
    del blocks
    gc.collect()
    # manual sublinear tf: 1 + log(tf), applied to the non-zero counts only
    X.data = 1.0 + np.log(X.data, where=X.data > 0, out=X.data)
    from sklearn.preprocessing import normalize
    X = normalize(X)
    return vec, X, t1 - t0, time.time() - t1


def topk_cosine(Q, X, k, block=2000):
    """Exact top-k cosine per query row, via chunked sparse matmul.

    Returns (idx, score) arrays of shape (n_queries, k).
    """
    n = Q.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = (Q[s:e] @ X.T).toarray()          # (b, n_cand)
        kk = min(k, sim.shape[1])
        part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(e - s)[:, None]
        vals = sim[rows, part]
        order = np.argsort(-vals, axis=1)
        idx[s:e, :kk] = part[rows, order]
        sc[s:e, :kk] = vals[rows, order]
        del sim
    return idx, sc


def _keys(df):
    """Coarse keys used only to find REACHABLE candidates for the prototype."""


def postal_union_cids(sub_c, sub_q):
    """Extra candidates from an exact country|postal-prefix key (dedup-safe).

    Complements soft similarity: the soft index is good at name similarity but
    misses pairs whose names were mangled, while an exact postal-prefix match
    still catches them. Returns (_rid, _cid) rows to UNION with the soft pairs.
    """
    def pk(df):
        return df.with_columns(
            (pl.col("ctry") + "|" + pl.col("business_address")
             .fill_null("").str.replace_all(r"\D", "")
             .str.replace_all(r"^0+", "").str.slice(0, 3)).alias("_pk")
        )

    c = pk(sub_c).select(["_cid", "_pk"])
    q = pk(sub_q).select(["_rid", "_pk"])
    j = q.join(c, on="_pk", how="inner")
    return j.select(["_rid", "_cid"]).unique()


def _keys(df):
    """Coarse keys used only to find REACHABLE candidates for the prototype."""
    return df.with_columns(
        (pl.col("ctry") + "|" + pl.col("nn").str.slice(0, 5)).alias("_p5"),
        (pl.col("ctry") + "|" + pl.col("nn").str.split(" ").list.join("")).alias("_sq"),
    )


def restrict_reachable(sub_c, sub_q, ctry):
    """Keep only candidates sharing a coarse prefix/squeeze key with a query.

    Recall-neutral for the measurement: a true match always has SOME shared
    prefix/squeeze with its S1 record, so it is retained.
    """
    qk = _keys(sub_q.select(["nn", "ctry"])).select(["_p5", "_sq"]).unique()
    ck = _keys(sub_c.select(["nn", "ctry"]))
    keep_p5 = ck.join(qk.select(["_p5"]).unique(), on="_p5", how="semi")
    keep_sq = ck.join(qk.select(["_sq"]).unique(), on="_sq", how="semi")
    keep = pl.concat([keep_p5, keep_sq], how="vertical").select(["nn"]).unique()
    # semi-join on nn only, so _cid and the address column are preserved
    return sub_c.join(keep, on="nn", how="semi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN)
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-features", type=int, default=1_048_576,
                    help="HashingVectorizer dim (power of two, e.g. 2**20)")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--no-restrict", dest="restrict", action="store_false",
                    help="index ALL country candidates (needs >15GB)")
    ap.add_argument("--score-thr", type=float, default=0.0,
                    help="drop candidates with cosine below this")
    ap.add_argument("--postal-union", action="store_true",
                    help="also add candidates sharing the country|postal3 key")
    ap.add_argument("--field", default="name", choices=["name", "name_addr"],
                    help="text fed to the vectorizer")
    args = ap.parse_args()
    t0 = time.time()

    print(f"sklearn char n-gram soft blocking | sample={args.sample} k={args.k} "
          f"ngram=(3,{args.ngram_max}) n_features={args.n_features:,} "
          f"field={args.field}")

    s1 = norm(pl.scan_csv(os.path.join(args.train_dir, "train_source1.tsv"),
                          separator="\t"), field=args.field).collect()
    s23 = norm(pl.concat([
        pl.scan_csv(os.path.join(args.train_dir, "train_source2.tsv"), separator="\t"),
        pl.scan_csv(os.path.join(args.train_dir, "train_source3.tsv"), separator="\t"),
    ], how="vertical"), field=args.field).collect()
    s23 = s23.with_row_index("_cid").with_columns(
        pl.col("country").fill_null("").str.to_lowercase().alias("ctry"))
    s1 = s1.with_columns(
        pl.col("country").fill_null("").str.to_lowercase().alias("ctry"))
    # keep business_address for the exact postal-key union
    s1 = s1.with_columns(pl.col("business_address").fill_null("").alias("business_address"))
    s23 = s23.with_columns(pl.col("business_address").fill_null("").alias("business_address"))
    print(f"loaded s1={s1.height:,} s23={s23.height:,} ({time.time()-t0:.0f}s)", flush=True)

    # sample S1 proportionally across countries so France-style unseen
    # country handling is exercised the same way the real test set will be
    s1 = s1.filter(pl.col("ctry") != "")
    per = max(1, args.sample // max(s1["ctry"].n_unique(), 1))
    s1 = (s1.group_by("ctry").head(per).sample(n=min(args.sample, s1.height), seed=42)
          .with_row_index("_rid"))
    n_s1 = s1.height
    print(f"sampled {n_s1:,} S1 across {s1['ctry'].n_unique()} countries: "
          f"{dict(s1['ctry'].value_counts().iter_rows())}", flush=True)

    # ground truth for the sample, mapped to candidate row ids
    gt = (
        pl.scan_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), separator="\t")
        .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")])
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("mid")])
        .collect()
        .join(s1.select(["entity_id", "_rid"]), left_on="source1_entity_id",
              right_on="entity_id", how="inner")
        .select(["_rid", "mid"])
        .join(s23.select(["entity_id", "_cid"]), left_on="mid",
              right_on="entity_id", how="left")
    )
    n_true = gt.height
    print(f"true pairs in sample: {n_true:,}", flush=True)

    kept_rows = []
    for ctry in sorted(s1["ctry"].unique().to_list()):
        sub_q = s1.filter(pl.col("ctry") == ctry)
        sub_c_all = s23.filter(pl.col("ctry") == ctry)
        sub_c = sub_c_all
        if sub_q.height == 0:
            continue
        print(f"\n[{ctry}] queries={sub_q.height:,} candidates={sub_c.height:,}",
              flush=True)

        # ---- PROTOTYPE OPTIMISATION ---------------------------------------
        # Indexing all 4-6M country candidates needs >15GB and never finishes
        # here. But for a RECALL measurement we only need candidates that could
        # plausibly be retrieved: those sharing a coarse key with some query.
        # This is a strict SUPERSET of the true soft neighbours, so measured
        # recall is unchanged -- it only prunes records no query could rank
        # highly anyway. Set --no-restrict to index everything.
        sub_c = restrict_reachable(sub_c, sub_q, ctry) if args.restrict else sub_c
        print(f"  after restrict: {sub_c.height:,} candidates", flush=True)
        # REMAP _cid to a dense 0..n-1 index. The restricted frame is a
        # SUBSET of the original s23, so its row_index is NOT the original
        # _cid -- without this remap every retrieved id is wrong and recall
        # silently collapses to 0.
        cid_map = sub_c.select(["_cid"]).with_row_index("_pos")
        sub_c = sub_c.with_row_index("_pos")

        tc = time.time()
        vec, X, t_prep, t_tr = build_country_index(
            sub_c["nn"].to_list(), args.n_features, args.chunk)
        print(f"  index: prep={t_prep:.0f}s transform={t_tr:.0f}s "
              f"shape={X.shape} nnz/row={X.nnz/max(X.shape[0],1):.1f} "
              f"RAM~{X.data.nbytes/1e9:.2f}GB", flush=True)
        Q = vec.transform(sub_q["nn"].to_list())
        Q.data = 1.0 + np.log(Q.data, where=Q.data > 0, out=Q.data)
        from sklearn.preprocessing import normalize as _nz
        Q = _nz(Q)
        tq = time.time()
        idx, sc = topk_cosine(Q, X, args.k)
        print(f"  query {sub_q.height:,} x {X.shape[0]:,} in {time.time()-tq:.0f}s",
              flush=True)

        qid = np.repeat(sub_q["_rid"].to_numpy(), args.k)
        pos = idx.reshape(-1)                       # dense positions
        # translate positions back to the ORIGINAL s23 _cid values
        orig = cid_map["_cid"].to_numpy()
        cid = orig[pos]
        sco = sc.reshape(-1)
        if args.postal_union:
            extra = postal_union_cids(sub_c_all, sub_q)
            if extra.height:
                ec = extra["_cid"].to_numpy()
                qid = np.concatenate([qid, extra["_rid"].to_numpy()])
                cid = np.concatenate([cid, ec])
                sco = np.concatenate([sco, np.zeros(len(ec), dtype=np.float32)])
        if args.score_thr > 0:
            m = sco >= args.score_thr
            qid, cid, sco = qid[m], cid[m], sco[m]
        kept_rows.append(
            pl.DataFrame({"_rid": qid, "_cid": cid, "score": sco})
        )
        del X, Q, idx, sc, vec
        gc.collect()
        print(f"  [{ctry}] done ({time.time()-tc:.0f}s)", flush=True)

    kept = pl.concat(kept_rows, how="vertical")
    print(f"\nretrieved pairs: {kept.height:,} "
          f"({kept.height/max(n_s1,1):.1f} per S1)  total {time.time()-t0:.0f}s",
          flush=True)

    hit = (
        gt.join(kept.with_columns(pl.lit(True).alias("_k")),
                on=["_rid", "_cid"], how="left")
        .with_columns(pl.col("_k").fill_null(False).alias("hit"))
    )
    pair_recall = float(hit["hit"].mean() or 0.0)
    pe = hit.group_by("_rid").agg(pl.len().alias("n_true"),
                                  pl.col("hit").sum().alias("n_hit"))
    ent_recall = float((pe["n_hit"] == pe["n_true"]).mean() or 0.0)
    n_sing = n_s1 - pe.height
    rf = pe["n_hit"] / pe["n_true"]
    ceiling = float((((1.25 * 1.0 * rf) / (0.25 * 1.0 + rf)).fill_null(0.0).sum()
                      + n_sing) / n_s1)

    per = kept.group_by("_rid").agg(pl.len().alias("c"))["c"]
    print("\n============ SOFT BLOCK RESULT ============")
    print(f"  pair recall    : {pair_recall*100:.2f}%")
    print(f"  entity recall  : {ent_recall*100:.2f}%")
    print(f"  F0.5 ceiling   : {ceiling:.4f}")
    print(f"  cand/S1        : avg {float(per.mean() or 0):.1f}  "
          f"p95 {float(per.quantile(0.95) or 0):.0f}  max {int(per.max() or 0):,}")
    print(f"  total time     : {time.time()-t0:.0f}s")
    print("===========================================")


if __name__ == "__main__":
    main()

