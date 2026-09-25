#!/usr/bin/env python3
"""
Submission 1 — Strict Exact-Match Baseline
Amazon ML Challenge 2026: Business Entity Resolution

This baseline implements a deterministic, conservative exact-match rule:
    normalized_country(S1) == normalized_country(S2/S3)
    AND
    normalized_business_name(S1) == normalized_business_name(S2/S3)

Produces:
- output/matching_results.tsv
- output/candidate_pairs.tsv
"""

import argparse
import csv
import os
import sys
import time
import unicodedata
import string
from collections import defaultdict
import numpy as np
import pandas as pd

# Precompute translation table for Unicode and ASCII punctuation replacement
PUNCTUATION_CHARS = {
    i: 32  # ord(' ')
    for i in range(0x110000)
    if unicodedata.category(chr(i)).startswith('P') or chr(i) in string.punctuation
}


def normalize_text(text: str) -> str | None:
    """Normalize business name or country string according to Submission 1 rules.
    
    1. Convert to Unicode casefold.
    2. Replace punctuation characters with spaces.
    3. Collapse consecutive whitespace to single space & strip leading/trailing spaces.
    4. Return None if missing, NaN, or empty.
    """
    if text is None or pd.isna(text):
        return None
    s = str(text).casefold().translate(PUNCTUATION_CHARS)
    s = " ".join(s.split())
    return s if s else None


def build_source_index(tsv_paths: list[str]) -> dict[tuple[str, str], list[str]]:
    """Build an index from (normalized_country, normalized_name) -> list of entity IDs."""
    index = defaultdict(list)
    total_records = 0
    valid_indexed = 0

    for path in tsv_paths:
        if not os.path.isfile(path):
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        print(f"Indexing records from {path}...")
        t0 = time.time()
        count = 0
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                count += 1
                total_records += 1
                eid = row.get("entity_id")
                bname = row.get("business_name")
                country = row.get("country")

                norm_name = normalize_text(bname)
                norm_country = normalize_text(country)

                # Null handling rule: if name or country is null, cannot be indexed/matched
                if norm_name is not None and norm_country is not None and eid:
                    index[(norm_country, norm_name)].append(eid)
                    valid_indexed += 1
        print(f"  Processed {count:,} records in {time.time() - t0:.2f}s")

    print(f"Total indexed records: {valid_indexed:,} across {len(index):,} unique keys")
    return index


def run_matching(
    test_dir: str, output_dir: str
) -> tuple[int, int, int, int]:
    """Run exact matching on test dataset and write matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(output_dir, exist_ok=True)
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.isfile(s1_path):
        raise FileNotFoundError(f"Source 1 test file not found: {s1_path}")

    # Step 1 & 2: Build index from S2 and S3
    index = build_source_index([s2_path, s3_path])

    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Processing Source 1 queries and generating output files...")
    t0 = time.time()

    total_s1 = 0
    s1_with_matches = 0
    total_candidate_links = 0

    with open(s1_path, mode="r", encoding="utf-8", newline="") as f_in, \
         open(matching_out_path, mode="w", encoding="utf-8", newline="") as f_match, \
         open(candidate_out_path, mode="w", encoding="utf-8", newline="") as f_cand:
        
        reader = csv.DictReader(f_in, delimiter="\t")
        
        # Write exact headers required
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in reader:
            total_s1 += 1
            s1_id = row.get("entity_id", "").strip()
            bname = row.get("business_name")
            country = row.get("country")

            norm_name = normalize_text(bname)
            norm_country = normalize_text(country)

            if norm_name is not None and norm_country is not None:
                matches = index.get((norm_country, norm_name), [])
                if matches:
                    # Deduplicate and sort lexicographically
                    unique_sorted_matches = sorted(list(dict.fromkeys(matches)))
                    match_str = ",".join(unique_sorted_matches)
                    s1_with_matches += 1
                    total_candidate_links += len(unique_sorted_matches)
                else:
                    match_str = ""
            else:
                match_str = ""

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{match_str}\n")

    s1_zero_matches = total_s1 - s1_with_matches
    print(f"Completed in {time.time() - t0:.2f}s")
    print(f"Wrote outputs to {matching_out_path} and {candidate_out_path}")
    print(f"Summary:")
    print(f"  Total S1 records: {total_s1:,}")
    print(f"  S1 records with >= 1 matches: {s1_with_matches:,}")
    print(f"  S1 records with 0 matches: {s1_zero_matches:,}")
    print(f"  Total candidate links: {total_candidate_links:,}")

    return total_s1, s1_with_matches, s1_zero_matches, total_candidate_links


def run_local_validation(train_dir: str):
    """Run local stratified 80/20 validation on training dataset."""
    print("=" * 60)
    print("RUNNING LOCAL VALIDATION ON TRAINING DATASET (80/20 Stratified Split)")
    print("=" * 60)
    
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading training data from {train_dir}...")
    t0 = time.time()
    train_s1 = pd.read_csv(s1_path, sep="\t")
    train_gt = pd.read_csv(gt_path, sep="\t")

    gt_dict = {}
    for s1_id, matches in zip(train_gt["source1_entity_id"], train_gt["matched_entity_ids"]):
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matches).strip().split(","))

    has_match = np.array([len(gt_dict.get(s1_id, set())) > 0 for s1_id in train_s1["entity_id"]], dtype=bool)

    # Fixed seed 42 stratified split
    rng = np.random.RandomState(42)
    indices_match = np.where(has_match)[0]
    indices_nomatch = np.where(~has_match)[0]
    rng.shuffle(indices_match)
    rng.shuffle(indices_nomatch)

    val_match_count = int(round(len(indices_match) * 0.2))
    val_nomatch_count = int(round(len(indices_nomatch) * 0.2))

    val_indices = np.concatenate([indices_match[:val_match_count], indices_nomatch[:val_nomatch_count]])
    val_indices.sort()
    train_indices = np.concatenate([indices_match[val_match_count:], indices_nomatch[val_nomatch_count:]])
    train_indices.sort()

    val_s1 = train_s1.iloc[val_indices]

    print(f"Validation Split Details:")
    print(f"  Train portion size: {len(train_indices):,} (has_match ratio: {has_match[train_indices].mean():.6f})")
    print(f"  Held-out 20% size:  {len(val_indices):,} (has_match ratio: {has_match[val_indices].mean():.6f})")

    # Index train S2 and S3
    index = build_source_index([s2_path, s3_path])

    print("Evaluating held-out validation set...")
    t_eval = time.time()
    precisions = []
    recalls = []
    correct_singletons = 0
    included_entities = 0

    for eid, name, country in zip(val_s1["entity_id"], val_s1["business_name"], val_s1["country"]):
        norm_name = normalize_text(name)
        norm_country = normalize_text(country)

        if norm_name is not None and norm_country is not None:
            preds = set(index.get((norm_country, norm_name), []))
        else:
            preds = set()

        true_matches = gt_dict.get(eid, set())

        tp = len(preds.intersection(true_matches))
        fp = len(preds - true_matches)
        fn = len(true_matches - preds)

        if len(true_matches) == 0 and len(preds) == 0:
            correct_singletons += 1
        else:
            included_entities += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    denom = 0.25 * macro_prec + macro_rec
    f05 = (1.25 * macro_prec * macro_rec) / denom if denom > 0 else 0.0

    print(f"Evaluation completed in {time.time() - t_eval:.2f}s")
    print(f"Validation Results:")
    print(f"  Total validation entities: {len(val_s1):,}")
    print(f"  Correctly-identified singletons (excluded from avg): {correct_singletons:,}")
    print(f"  Entities included in macro average: {included_entities:,}")
    print(f"  Macro Precision: {macro_prec:.6f}")
    print(f"  Macro Recall:    {macro_rec:.6f}")
    print(f"  F0.5 Score:      {f05:.6f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Submission 1 — Strict Exact-Match Baseline for Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Directory containing training files (if --validate is enabled)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run local 80/20 stratified validation on training data before test matching",
    )

    args = parser.parse_args()

    if args.validate:
        train_dir = args.train_dir or os.path.join(os.path.dirname(args.test_dir), "train")
        run_local_validation(train_dir)

    print("\n" + "=" * 60)
    print("RUNNING TEST INFERENCE — EXACT MATCH BASELINE")
    print("=" * 60)
    run_matching(args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()
