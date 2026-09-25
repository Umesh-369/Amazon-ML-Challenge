#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Precision-First Pipeline (v03)

Architecture:
1. Multi-Channel Candidate Generation (Blocking):
   - Channel 1: Exact Normalized Business Name in Country
   - Channel 2: Core Business Name (stripped legal suffixes) in Country
   - Channel 3: Primary Building Number + Distinct Name Token
   - Channel 4: Primary Building Number + Distinct Name 4-char Prefix
   - Channel 5: Primary Building Number + Distinct Address Token (Street/Area)
   - Channel 6: First Name Token + First Address Token

2. Precision-First Scoring & Disambiguation:
   - Name Similarity: max(ratio, token_set_ratio) via rapidfuzz
   - Address Similarity: token_set_ratio via rapidfuzz
   - Address Number Agreement: exact match bonus (+10), strict conflict penalty (0 score)
   - Cross-Lingual / Transliteration Anchor: high address similarity + building number match
   - Operating Threshold: score >= 90.0 for final match decision
   - Candidate pairs: all blocked candidates emitted to candidate_pairs.tsv
   - Final matches: high-precision matches emitted to matching_results.tsv (subset of candidates)
"""

import argparse
import csv
import os
import re
import string
import sys
import time
import unicodedata
from collections import defaultdict
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

# Fast C Translation table for Unicode casefolding, punctuation removal, and diacritic normalization
CHAR_MAP = {ord(c): 32 for c in string.punctuation}
for i in range(0x110000):
    if unicodedata.category(chr(i)).startswith('P'):
        CHAR_MAP[i] = 32

diacritic_pairs = 'àa áa âa ãa äa åa èe ée êe ëe ìi íi îi ïi òo óo ôo õo öo ùu úu ûu üu ýy ÿy çc ñn'
for pair in diacritic_pairs.split():
    CHAR_MAP[ord(pair[0])] = ord(pair[1])
    CHAR_MAP[ord(pair[0].upper())] = ord(pair[1])


def fast_norm(text: str) -> str:
    """Fast C-level text normalization handling punctuation, whitespace, and diacritics."""
    if not text:
        return ""
    return " ".join(str(text).casefold().translate(CHAR_MAP).split())


LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'co', 'corp', 'corporation',
    'company', 'enterprises', 'enterprise', 'services', 'service', 'solutions', 'solution',
    'technologies', 'technology', 'group', 'holdings', 'llp', 'pllc', 'sa', 'sarl', 'sas'
}

STOP_WORDS = {
    'the', 'and', 'of', 'in', 'at', 'on', 'near', 'opp', 'opposite', 'behind', 'floor',
    'road', 'street', 'st', 'rd', 'ave', 'avenue', 'lane', 'drive', 'dr', 'way', 'hwy', 'highway',
    'plot', 'block', 'sector', 'nagar', 'colony', 'apartment', 'apartments', 'bldg', 'building',
    'rue', 'avenue', 'boulevard', 'bd', 'chemin', 'place'
}


def extract_tokens(text: str, min_len=3):
    norm_s = fast_norm(text)
    return [t for t in norm_s.split() if len(t) >= min_len and t not in STOP_WORDS and t not in LEGAL_SUFFIXES]


def extract_numbers(addr: str):
    if not addr:
        return []
    nums = re.findall(r'\b\d+\b', addr)
    res = []
    for n in nums:
        if 0 < int(n) < 10000000:
            res.append(int(n))
    return res[:2]


def generate_blocking_keys(country: str, name: str, addr: str):
    c = fast_norm(country)
    n_norm = fast_norm(name)
    n_tokens = extract_tokens(name, min_len=3)
    a_nums = extract_numbers(addr)
    a_tokens = extract_tokens(addr, min_len=3)

    keys = set()
    # 1. Exact Name
    if n_norm:
        keys.add(('EXACT', c, n_norm))
    # 2. Core Name (sorted tokens without legal suffixes)
    if len(n_tokens) >= 2:
        keys.add(('CORE', c, " ".join(sorted(n_tokens[:4]))))
    # 3. Name token + building number
    if a_nums:
        p_num = a_nums[0]
        for t in n_tokens[:2]:
            keys.add(('NAME_NUM', c, t, p_num))
            if len(t) >= 4:
                keys.add(('PREF_NUM', c, t[:4], p_num))
        # 4. Address number + street token
        for at in a_tokens[:2]:
            keys.add(('ADDR_NUM_TOK', c, p_num, at))
    # 5. First name token + distinct address token
    if n_tokens and a_tokens:
        keys.add(('NAME_ADDR', c, n_tokens[0], a_tokens[0]))

    return keys


def compute_pair_score(s1_name, s1_addr, s1_nums, c_name, c_addr, c_nums):
    name_ratio = fuzz.ratio(s1_name, c_name)
    name_token_set = fuzz.token_set_ratio(s1_name, c_name)
    best_name_sim = max(name_ratio, name_token_set)

    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr)

    has_both_nums = len(s1_nums) > 0 and len(c_nums) > 0
    num_match = len(s1_nums & c_nums) > 0 if has_both_nums else False
    num_conflict = (has_both_nums and not num_match)

    if num_conflict:
        return 0.0

    score = 0.65 * best_name_sim + 0.35 * addr_token_set
    if num_match:
        score += 10.0

    # Cross-lingual / Transliterated / Acronym anchor
    if addr_token_set >= 85 and num_match:
        score = max(score, 90.0)

    return min(100.0, score)


def run_pipeline(source1_path: str, source2_path: str, source3_path: str,
                 matching_out: str, candidate_out: str,
                 match_threshold: float = 90.0):
    print("=" * 70)
    print("RUNNING PRECISION-FIRST ENTITY RESOLUTION PIPELINE")
    print(f"  Source 1: {source1_path}")
    print(f"  Source 2: {source2_path}")
    print(f"  Source 3: {source3_path}")
    print(f"  Threshold: {match_threshold}")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Read all S1 queries and collect needed blocking keys
    print("Step 1: Reading Source 1 queries and generating query blocking keys...")
    t0 = time.time()
    s1_rows = []
    needed_keys = set()
    with open(source1_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_rows.append(row)
            keys = generate_blocking_keys(row.get('country'), row.get('business_name'), row.get('business_address'))
            needed_keys.update(keys)

    print(f"  Loaded {len(s1_rows):,} Source 1 records.")
    print(f"  Generated {len(needed_keys):,} unique query blocking keys in {time.time()-t0:.1f}s.")

    # Step 2: Stream S2 & S3 and build inverted index only for needed keys
    print("\nStep 2: Indexing Source 2 & Source 3 against query keys...")
    t0 = time.time()
    inverted_index = defaultdict(list)
    s23_records = {}
    total_s23 = 0

    for path in [source2_path, source3_path]:
        t_file = time.time()
        c = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                c += 1
                total_s23 += 1
                keys = generate_blocking_keys(row.get('country'), row.get('business_name'), row.get('business_address'))
                matching = keys & needed_keys
                if matching:
                    eid = row['entity_id']
                    s23_records[eid] = (
                        fast_norm(row.get('business_name')),
                        fast_norm(row.get('business_address')),
                        set(extract_numbers(row.get('business_address')))
                    )
                    for k in matching:
                        if len(inverted_index[k]) < 50:
                            inverted_index[k].append(eid)
        print(f"  Processed {c:,} records in {path} in {time.time()-t_file:.1f}s")

    print(f"  Indexing complete in {time.time()-t0:.1f}s.")
    print(f"  Active keys: {len(inverted_index):,}, Cached candidate records: {len(s23_records):,}")

    # Step 3: Candidate Scoring & Output Generation
    print("\nStep 3: Scoring candidates and generating submission outputs...")
    t0 = time.time()
    os.makedirs(os.path.dirname(matching_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(candidate_out) or '.', exist_ok=True)

    total_s1 = 0
    matched_s1 = 0
    total_candidates = 0
    total_matches = 0

    with open(matching_out, 'w', encoding='utf-8', newline='') as f_match, \
         open(candidate_out, 'w', encoding='utf-8', newline='') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in s1_rows:
            total_s1 += 1
            s1_id = row['entity_id']
            s1_name = fast_norm(row.get('business_name'))
            s1_addr = fast_norm(row.get('business_address'))
            s1_nums = set(extract_numbers(row.get('business_address')))

            keys = generate_blocking_keys(row.get('country'), row.get('business_name'), row.get('business_address'))
            cands = set()
            for k in keys:
                if k in inverted_index:
                    cands.update(inverted_index[k])

            total_candidates += len(cands)

            # Score each candidate
            matched_ids = []
            for cid in cands:
                c_data = s23_records.get(cid)
                if not c_data:
                    continue
                c_name, c_addr, c_nums = c_data
                score = compute_pair_score(s1_name, s1_addr, s1_nums, c_name, c_addr, c_nums)
                if score >= match_threshold:
                    matched_ids.append((score, cid))

            # Group candidates by source (S2 vs S3) sorted by score descending
            # Allow top-3 per source
            s2_cands = sorted([x for x in matched_ids if x[1].startswith('S2-')], key=lambda x: x[0], reverse=True)
            s3_cands = sorted([x for x in matched_ids if x[1].startswith('S3-')], key=lambda x: x[0], reverse=True)
            top_s2 = [cid for sc, cid in s2_cands[:3]]
            top_s3 = [cid for sc, cid in s3_cands[:3]]
            final_matches = sorted(list(set(top_s2 + top_s3)))

            # Candidates list: sorted lexicographically
            cand_str = ",".join(sorted(list(cands)))
            match_str = ",".join(final_matches)

            if final_matches:
                matched_s1 += 1
                total_matches += len(final_matches)

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            if total_s1 % 200000 == 0:
                print(f"  Processed {total_s1:,}/{len(s1_rows):,} queries in {time.time()-t0:.1f}s...")

    print(f"\nExecution finished in {time.time()-t_start:.1f}s.")
    print(f"Summary:")
    print(f"  Total S1 entities: {total_s1:,}")
    print(f"  S1 with >= 1 match: {matched_s1:,} ({matched_s1/total_s1*100:.2f}%)")
    print(f"  Singletons: {total_s1 - matched_s1:,} ({(total_s1 - matched_s1)/total_s1*100:.2f}%)")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Total final matches:   {total_matches:,}")
    print(f"  Outputs saved to:\n    {matching_out}\n    {candidate_out}")


def main():
    parser = argparse.ArgumentParser(description="Precision-First Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Test directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=90.0, help="Matching score threshold")
    args = parser.parse_args()

    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")
    match_out = os.path.join(args.output_dir, "matching_results.tsv")
    cand_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    run_pipeline(s1_path, s2_path, s3_path, match_out, cand_out, match_threshold=args.threshold)


if __name__ == "__main__":
    main()
