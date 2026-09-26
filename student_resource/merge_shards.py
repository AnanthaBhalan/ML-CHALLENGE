"""
Merge sharded inference output back into the two submission TSVs.

Shard files are concatenated IN SHARD ORDER so the final file is in the same
row order as test_source1.tsv. Hard-fails on any duplicate or missing S1 id,
because a duplicate row in matching_results.tsv is a submission error.

    python merge_shards.py --total 1732545
"""

import argparse
import os
import sys

OUT = r"c:\Users\admin\Desktop\ML CHALLENGE\student_resource\output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--total", type=int, required=True,
                    help="expected number of S1 rows in test_source1.tsv")
    ap.add_argument("--shards", required=True,
                    help="comma list of suffixes in order, e.g. _s0,_s1,_s2 "
                         "(prefix 'base' inserts the pre-existing unsuffixed file)")
    args = ap.parse_args()

    for stem, col in (("matching_results", "matched_entity_ids"),
                      ("candidate_pairs", "candidate_entity_ids")):
        hdr = f"source1_entity_id\t{col}"
        out_path = os.path.join(args.out_dir, f"{stem}.tsv")
        parts = []
        for sfx in args.shards.split(","):
            sfx = sfx.strip()
            if sfx == "base":
                p = os.path.join(args.out_dir, f"{stem}.tsv")
            else:
                p = os.path.join(args.out_dir, f"{stem}{sfx}.tsv")
            if not os.path.exists(p):
                print(f"[ABORT] missing shard file: {p}")
                return 1
            parts.append((sfx, p))
            print(f"  part: {os.path.basename(p)}  "
                  f"{max(sum(1 for _ in open(p, encoding='utf-8')) - 1, 0):,} rows")

        seen = set()
        n = 0
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(hdr + "\n")
            for sfx, p in parts:
                with open(p, encoding="utf-8") as fh:
                    fh.readline()  # drop this part's header
                    for line in fh:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        eid = line.split("\t", 1)[0]
                        if eid in seen:
                            print(f"[ABORT] DUPLICATE s1 id {eid} (in {p})")
                            return 1
                        seen.add(eid)
                        out.write(line + "\n")
                        n += 1
        print(f"\n{stem}: wrote {n:,} rows (expected {args.total:,})")
        if n != args.total:
            print(f"[ABORT] row count mismatch")
            return 1
    print("\n[ok] merge complete, no duplicates, counts exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
