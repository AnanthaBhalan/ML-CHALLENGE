"""
Diagnose WHY name_p4 produces ~7,400 candidates per S1 entity.

Hypothesis under test: a handful of mega-buckets absorb a large share of
records, so a few leading noise tokens (m/s, --, <<, shri, ...) funnel
thousands of unrelated businesses into one block.

Outputs:
  * distinct-key count vs record count (the implied average bucket size)
  * cumulative share of S2/S3 records captured by the top-N buckets
  * share of S1 entities that land in a mega-bucket
  * raw-name examples from the worst buckets
"""

import os
import sys

import polars as pl

TRAIN = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\dataset\train"


def scan(s):
    return pl.scan_csv(os.path.join(TRAIN, f"train_source{s}.tsv"), separator="\t")


def norm(df):
    """Lowercase, punctuation -> space, collapse whitespace."""
    return df.with_columns(
        pl.col("business_name")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("nn")
    )


def add_key(df):
    """Reproduce the exact name_p4 key: country | first-4-of-first-token."""
    return (
        norm(df)
        .with_columns(
            pl.col("nn")
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .list.first()
            .alias("ft")
        )
        .with_columns(
            (pl.col("country").fill_null("").str.to_lowercase() + "|" + pl.col("ft").fill_null("").str.slice(0, 4)).alias("k")
        )
    )


def main():
    frames = {}
    for s in (1, 2, 3):
        frames[s] = add_key(scan(s)).select(
            "entity_id", "business_name", "country", "nn", "ft", "k"
        ).collect()
        n = frames[s].height
        empty_nn = int((frames[s]["nn"].str.len_chars() == 0).sum())
        null_ft = int(frames[s]["ft"].is_null().sum())
        print(
            f"source{s}: rows={n:>10,}  nn_empty={empty_nn:>9,} "
            f"({empty_nn/n*100:5.2f}%)  first_token_null={null_ft:>9,} ({null_ft/n*100:5.2f}%)"
        )

    s23 = pl.concat([frames[2], frames[3]], how="vertical")
    total = s23.height
    print(f"\nS2+S3 combined: {total:,} records")

    # ---- bucket distribution ------------------------------------------------
    buckets = (
        s23.group_by("k")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .with_columns((pl.col("n").cum_sum() / total).alias("cum_share"))
    )
    distinct = buckets.height
    print(f"distinct name_p4 keys: {distinct:,}")
    print(f"implied average bucket size: {total/distinct:,.1f}")

    print("\nTop 25 buckets (n = records, cum = share of ALL S2/S3 records up to here):")
    top = buckets.head(25)
    for r in top.iter_rows(named=True):
        print(f"   n={r['n']:>10,}  cum={r['cum_share']*100:6.2f}%  key={r['k']!r}")

    for N in (10, 50, 100, 500, 1000, 5000):
        if N <= distinct:
            cum_at = buckets["cum_share"][N - 1]
            print(f"top {N:>5} buckets cover {cum_at*100:6.2f}% of S2/S3 records")

    # ---- how much of S1 is poisoned by mega-buckets -------------------------
    for thr in (1000, 10000, 100000):
        mega = buckets.filter(pl.col("n") >= thr).select("k")
        hit = frames[1].join(mega, on="k", how="semi").height
        print(
            f"S1 entities in buckets with n>={thr:>7,}: {hit:>10,} "
            f"({hit/frames[1].height*100:5.2f}% of S1)"
        )

    # ---- raw examples from the worst buckets --------------------------------
    worst = buckets.head(12)["k"].to_list()
    print("\nRaw business_name examples from the 12 worst buckets:")
    ex = (
        s23.filter(pl.col("k").is_in(worst))
        .group_by("k")
        .agg(pl.col("business_name").head(5).alias("names"), pl.len().alias("n"))
        .sort("n", descending=True)
    )
    for r in ex.iter_rows(named=True):
        print(f"\n  key={r['k']!r}  n={r['n']:,}")
        for nm in r["names"]:
            print(f"      {nm!r}")

    # ---- most common first tokens overall ------------------------------------
    print("\nTop 30 first-tokens across S2+S3 (candidates for stopword removal):")
    ftc = (
        s23.filter(pl.col("ft").is_not_null())
        .group_by("ft")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(30)
    )
    for r in ftc.iter_rows(named=True):
        print(f"   {r['ft']!r:<28} {r['n']:>10,}")


if __name__ == "__main__":
    main()
