"""
Volume-crushing levers, measured honestly.

LEVER 1  exact out-of-core IDF / Zipfian stop-token annihilation
LEVER 2  intersected postal key (country | postal | first-2 name chars)
LEVER 3  hard cap per S1

Order of operations that actually works:
  1. count candidates per KEY (cheap, O(records))
  2. only join the keys whose bucket is under a size threshold
  3. cap per S1 only when materialising the final shortlist

WHY THE NAIVE `.head(1000)` INSIDE group_by().agg() IS NOT AN OOM FIX
  Polars must EXECUTE the join before it can aggregate. For a loose key that
  join is ~10^10 rows; the cap then discards almost all of them, but only
  after the peak memory has already been hit. A cap limits OUTPUT size, not
  peak memory. The correct order is: count per KEY (cheap) -> decide which
  keys are safe -> join only those -> cap per S1 when writing the shortlist.
  Every number below is measured WITHOUT materialising the pair set.

MEASURED RESULTS (train, 10,320,219 S2/S3 docs, ~13s total)
  LEVER 1 exact IDF: 52 tokens appear in >=0.5% of documents, covering 92.2%
          of all documents: limited, private, llc, ltd, inc, &, center, pvt,
          partners, services, group, co, corp, and, holdings, care, llp,
          associates, corporation, enterprises, service...
          1,836,966 informative tokens kept. Works as intended and is cheap,
          but note these are the SAME legal-form tokens the hand-written
          stopword list already removed -- it is confirmatory, not new.

  LEVER 2 intersected key, postal REQUIRED (the proposed config):
          only 7.7% of S2/S3 and 6.8% of S1 have a 5-6 digit postal
          avg 2.60 cand/S1, MAX 21, 390,348 pairs -- but this covers just
          149,948 of 2,206,821 S1 entities (6.8%)
          pair recall 77.90% but ENTITY recall 1.92%, ceiling 0.1077
          => the 92.3% of entities with no extractable postal get NOTHING.

  LEVER 2 with null postal allowed (postal = "" bucket):
          avg 39,422 cand/S1, MAX 174,826, 87.0 BILLION pairs
          pair recall 75.03%, entity recall 45.59%, ceiling 0.8853

BOTTOM LINE
  The intersected key does NOT deliver 50-150 candidates at high recall.
  It is bimodal: either a tiny unusable sample (postal present) or a 39k
  blow-up (postal absent). The expected 50-150 figure is an artefact of
  measuring only the 7.7% of rows that HAVE a postal code.

  Root cause: `(\\d{5,6})` matches a CONTIGUOUS 5-6 digit run. Real addresses
  are mostly "1795 Westchester Drive" (4-digit street number), Indian
  "G-3/571" or "570/13" (split by punctuation), and landmark text. Only
  7.7% yield a match. A digit-run extractor that also handles split codes
  would raise coverage, but coverage is necessary, not sufficient.

  Net: levers 1-3 as specified do not break the recall/volume wall. The
  best pool remains soft(name+address) + postal union from soft_block_proto
  (95.48% pair recall, 0.9861 ceiling, 25,684 cand/S1). A real ANN index
  (hnswlib/FAISS under Python <=3.12) plus a learned classifier is still
  the right architecture; the classifier is what converts the 0.9861
  ceiling into score.
"""

import argparse
import os
import time

import numpy as np
import polars as pl

TRAIN = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\train"


def norm_name(df):
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
            pl.col("nn").str.split(" ").list.first().alias("w1")
        )
    )


def add_keys(df):
    """LEVER 2: country | postal | clean first-2 name chars."""
    return norm_name(df).with_columns(
        pl.col("w1").str.slice(0, 2).alias("clean_p2"),
        pl.col("business_address")
        .fill_null("")
        .str.extract(r"(\d{5,6})", 1)
        .alias("postal"),
    ).with_columns(
        (pl.col("country").fill_null("")
         + "|" + pl.col("postal").fill_null("")
         + "|" + pl.col("clean_p2").fill_null("")).alias("ipk")
    )



def lever1_idf(s2, s3, thresh=0.005, top=25):
    """LEVER 1: exact document frequency over S2+S3, drop Zipfian tokens.

    Exact, not sampled. Returns (kept_idf_frame, dropped_tokens_frame).
    """
    t0 = time.time()
    corpus = pl.concat([
        s2.select(pl.col("business_name").fill_null("").str.to_lowercase().alias("name")),
        s3.select(pl.col("business_name").fill_null("").str.to_lowercase().alias("name")),
    ])
    n_docs = corpus.select(pl.len()).collect().item()
    print(f"[IDF] corpus documents: {n_docs:,}")

    tok = (
        corpus.with_columns(pl.col("name").str.split(" ").alias("t"))
        .explode("t", empty_as_null=True)
        .filter(pl.col("t").str.len_chars() > 0)
    )
    # n_unique docs per token => true DOCUMENT frequency (not token slots)
    dfc = tok.group_by("t").agg(pl.col("name").n_unique().alias("df"))
    cut = n_docs * thresh
    dropped = dfc.filter(pl.col("df") >= cut).sort("df", descending=True).collect()
    kept = (
        dfc.filter(pl.col("df") < cut)
        .with_columns((np.log((n_docs + 1) / (pl.col("df") + 1)) + 1).alias("idf"))
        .collect()
    )
    cov_drop = float(dropped["df"].sum()) / n_docs
    print(f"[IDF] dropped {dropped.height:,} tokens (in >={thresh:.1%} of docs), "
          f"present in {cov_drop:.2%} of all documents")
    print(f"[IDF] kept {kept.height:,} informative tokens  ({time.time()-t0:.0f}s)")
    top_names = [r[0] for r in dropped.head(top).iter_rows()]
    print("[IDF] top dropped (ascii-safe):",
          [t.encode("ascii", "replace").decode() for t in top_names])
    return kept, dropped


def lever2_measure(s1k, s23k, require_postal=True):
    """Count candidates per S1 for the intersected key WITHOUT joining.

    Mirrors blocking_eda.py's trick: count rows per key, then join the counts
    onto S1. O(records) instead of O(pairs), so a 10^10-pair explosion is
    never materialised.
    """
    a = s1k
    b = s23k
    if require_postal:
        a = a.filter(pl.col("postal").is_not_null())
        b = b.filter(pl.col("postal").is_not_null())
    cnt = b.group_by("ipk").agg(pl.len().alias("n"))
    per = (a.select(["entity_id", "ipk"]).join(cnt, on="ipk", how="left")
             .with_columns(pl.col("n").fill_null(0).cast(pl.UInt64)))
    c = per["n"]
    n1 = per.height
    tot = int(c.sum() or 0)
    nz = per.filter(pl.col("n") > 0).height
    print(f"\n[INTERSECTED KEY] S1 with non-null postal: {n1:,}")
    print(f"  avg candidates/S1 : {float(c.mean() or 0):,.2f}")
    print(f"  median           : {float(c.median() or 0):,.0f}")
    print(f"  p95              : {float(c.quantile(0.95) or 0):,.0f}")
    print(f"  p99              : {float(c.quantile(0.99) or 0):,.0f}")
    print(f"  MAX              : {int(c.max() or 0):,}")
    print(f"  zero-candidate   : {(n1 - nz)/max(n1,1)*100:.2f}%")
    print(f"  TOTAL PAIRS      : {tot:,}  ({tot/1e9:.2f}B)")
    return per



def lever2_recall(s1k, s23k, gt, require_postal=True):
    """Pair/entity recall of the intersected key, measured without exploding."""
    if require_postal:
        s1sel = s1k.filter(pl.col("postal").is_not_null()).select(["entity_id", "ipk"])
        s23sel = s23k.filter(pl.col("postal").is_not_null()).select(["entity_id", "ipk"])
    else:
        s1sel = s1k.select(["entity_id", "ipk"])
        s23sel = s23k.select(["entity_id", "ipk"])
    hit = (
        gt.join(s1sel.rename({"entity_id": "source1_entity_id", "ipk": "k1"}),
                on="source1_entity_id", how="left")
          .join(s23sel.rename({"entity_id": "mid", "ipk": "k2"}), on="mid", how="left")
          .with_columns((pl.col("k1") == pl.col("k2")).alias("hit"))
    )
    pair_recall = float(hit["hit"].mean() or 0.0)
    pe = hit.group_by("source1_entity_id").agg(
        pl.len().alias("n_true"), pl.col("hit").sum().alias("n_hit"))
    ent = float((pe["n_hit"] == pe["n_true"]).mean() or 0.0)
    n_sing = s1k.height - pe.height
    rf = pe["n_hit"] / pe["n_true"]
    ceil = float((((1.25 * 1.0 * rf) / (0.25 * 1.0 + rf)).fill_null(0.0).sum()
                   + n_sing) / max(s1k.height, 1))
    print(f"\n[RECALL require_postal={require_postal}] "
          f"pair={pair_recall*100:.2f}%  entity={ent*100:.2f}%  ceiling={ceil:.4f}")
    return pair_recall, ent, ceil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN)
    ap.add_argument("--idf-thresh", type=float, default=0.005)
    ap.add_argument("--out", default="exact_idf_weights.parquet")
    ap.add_argument("--no-postal", action="store_true",
                    help="also measure the key with postal allowed to be null")
    args = ap.parse_args()
    t0 = time.time()

    s1 = pl.scan_csv(os.path.join(args.train_dir, "train_source1.tsv"), separator="\t")
    s2 = pl.scan_csv(os.path.join(args.train_dir, "train_source2.tsv"), separator="\t")
    s3 = pl.scan_csv(os.path.join(args.train_dir, "train_source3.tsv"), separator="\t")
    gt = (
        pl.scan_csv(os.path.join(args.train_dir, "train_ground_truth.tsv"), separator="\t")
        .select(["source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("m")])
        .explode("m", empty_as_null=True)
        .filter(pl.col("m").is_not_null() & (pl.col("m").str.len_chars() > 0))
        .select(["source1_entity_id", pl.col("m").alias("mid")])
        .collect()
    )
    print(f"ground-truth pairs: {gt.height:,}  ({time.time()-t0:.0f}s)\n")

    # ---- LEVER 1 -----------------------------------------------------------
    kept, dropped = lever1_idf(s2, s3, thresh=args.idf_thresh)
    kept.write_parquet(args.out)
    print(f"[IDF] wrote {args.out}")

    # ---- LEVER 2 -----------------------------------------------------------
    s1k = add_keys(s1).collect()
    s23k = pl.concat([add_keys(s2), add_keys(s3)], how="vertical").collect()
    print(f"\n[KEYS] s1={s1k.height:,} s23={s23k.height:,}")
    n_post = int(s23k["postal"].is_not_null().sum())
    print(f"[KEYS] S2/S3 rows with an extractable 5-6 digit postal: "
          f"{n_post:,} ({n_post/max(s23k.height,1)*100:.1f}%)")
    s1_post = int(s1k["postal"].is_not_null().sum())
    print(f"[KEYS] S1 rows with an extractable 5-6 digit postal: "
          f"{s1_post:,} ({s1_post/max(s1k.height,1)*100:.1f}%)")

    lever2_measure(s1k, s23k, require_postal=True)
    lever2_recall(s1k, s23k, gt, require_postal=True)
    if args.no_postal:
        lever2_measure(s1k, s23k, require_postal=False)
        lever2_recall(s1k, s23k, gt, require_postal=False)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
