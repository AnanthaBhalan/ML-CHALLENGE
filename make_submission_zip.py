"""
Package the ML Challenge submission zip.

Run ONLY after test inference has completed (both output TSVs must have the
full 1,732,545 rows) and after utils/validate_submission.py passes.

    python make_submission_zip.py

Produces  submission.zip  containing:
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/src/*.py
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    code/Documentation.md

It REFUSES to build if validation fails or row counts are short, so a
malformed package cannot be shipped by accident.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"c:\Users\admin\Desktop\ML CHALLENGE")
RES = ROOT / "student_resource"
OUT = RES / "output"                      # where inference writes
PKG = ROOT / "submission_package"         # the tree we ship
PKG_OUT = PKG / "output"
PKG_CODE = PKG / "code"
ZIP = ROOT / "submission.zip"

EXPECTED_S1 = 1_732_545


def fail(msg):
    print(f"\n[ABORT] {msg}")
    sys.exit(1)


def count_rows(p: Path) -> int:
    with p.open("rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)  # minus header


def main():
    # ---- 1. both files must exist -------------------------------------
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = OUT / name
        if not p.exists():
            fail(f"missing {p} - is inference still running?")
        if p.stat().st_size == 0:
            fail(f"{name} is empty (0 bytes)")

    # ---- 2. row counts must be complete --------------------------------
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        n = count_rows(OUT / name)
        print(f"{name}: {n:,} rows (expected {EXPECTED_S1:,})")
        if n != EXPECTED_S1:
            fail(f"{name} has {n:,} rows, expected {EXPECTED_S1:,} "
                 f"- inference incomplete ({100.0 * n / EXPECTED_S1:.1f}%)")

    # ---- 3. run the official format validator --------------------------
    print("\n[validate] running validate_submission.py ...")
    r = subprocess.run(
        [sys.executable, str(RES / "utils" / "validate_submission.py"),
         "--matching", str(OUT / "matching_results.tsv"),
         "--candidates", str(OUT / "candidate_pairs.tsv")],
        capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.stderr.strip():
        print(r.stderr[-2000:])
    if r.returncode != 0:
        fail(f"validate_submission.py returned {r.returncode} - fix before packaging")

    # ---- 4. required code artefacts -------------------------------------
    required = [
        PKG_CODE / "business_entity_resolution" / "src" / "clf_pipeline.py",
        PKG_CODE / "business_entity_resolution" / "src" / "clf_infer.py",
        PKG_CODE / "business_entity_resolution" / "README.md",
        PKG_CODE / "business_entity_resolution" / "requirements.txt",
        PKG / "Documentation_template.md",
    ]
    for p in required:
        if not p.exists():
            fail(f"missing code artefact: {p}")
    print("\n[ok] all code artefacts present")

    # ---- 5. copy validated outputs into the package tree -----------------
    PKG_OUT.mkdir(parents=True, exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(OUT / name, PKG_OUT / name)
        print(f"[copy] {name} -> submission_package/output/")

    # ---- 6. build the zip ------------------------------------------------
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(PKG).as_posix())

    print(f"\n[done] {ZIP}  ({ZIP.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()
