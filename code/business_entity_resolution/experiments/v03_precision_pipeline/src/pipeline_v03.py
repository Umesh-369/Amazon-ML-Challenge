#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Pipeline — Ultra-Fast Precision Architecture (v03)

Pipeline Overview:
- Fast C-level normalization with universal diacritic and punctuation handling.
- Multi-Index Candidate Generation:
    1. (country, normalized_business_name)
    2. (country, core_business_name) [legal suffixes stripped]
    3. (country, building_number, street_token) [recovers transliterations & acronyms]
- Precision-First Disambiguation Logic:
    - Address number matching / conflict detection
    - Adaptive chain / duplicate name suppression
    - Source-aware top-match capping (max 2 per source)
- Generates both required files:
    output/matching_results.tsv
    output/candidate_pairs.tsv
- Fully passes utils/validate_submission.py
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

sys.stdout.reconfigure(encoding='utf-8')

# Fast C Translation Table
CHAR_MAP = {ord(c): 32 for c in string.punctuation}
for i in range(0x110000):
    if unicodedata.category(chr(i)).startswith('P'):
        CHAR_MAP[i] = 32

diacritic_pairs = 'àa áa âa ãa äa åa èe ée êe ëe ìi íi îi ïi òo óo ôo õo öo ùu úu ûu üu ýy ÿy çc ñn'
for pair in diacritic_pairs.split():
    CHAR_MAP[ord(pair[0])] = ord(pair[1])
    CHAR_MAP[ord(pair[0].upper())] = ord(pair[1])


def fast_norm(text: str) -> str:
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


def extract_core_name(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def extract_primary_number(addr: str) -> int:
    if not addr:
        return 0
    m = re.search(r'\b\d+\b', addr)
    if m:
        val = int(m.group(0))
        return val if 0 < val < 10000000 else 0
    return 0


def extract_first_street_token(norm_addr: str) -> str:
    for t in norm_addr.split():
        if len(t) >= 4 and t not in STOP_WORDS and not t.isdigit():
            return t
    return ""


def run_pipeline(test_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE (v03)")
    print(f"  Test dir:   {test_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    t_total = time.time()

    # Step 1: Index Source 2 and Source 3
    print("Step 1: Indexing Source 2 & Source 3...")
    t0 = time.time()
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_addr = defaultdict(list)

    total_s23 = 0
    for path in [s2_path, s3_path]:
        t_f = time.time()
        c = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                c += 1
                total_s23 += 1
                eid = row.get('entity_id')
                if not eid:
                    continue
                country = fast_norm(row.get('country'))
                bname = row.get('business_name')
                addr = row.get('business_address') or ""

                norm_name = fast_norm(bname)
                if not norm_name or not country:
                    continue

                b_num = extract_primary_number(addr)
                norm_addr = fast_norm(addr)

                # 1. Exact Name
                idx_exact[(country, norm_name)].append((eid, b_num))

                # 2. Core Name (without legal suffixes)
                core_name = extract_core_name(norm_name)
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num))

                # 3. Address Key (building_num + street_token)
                if b_num > 0:
                    s_tok = extract_first_street_token(norm_addr)
                    if s_tok:
                        idx_addr[(country, b_num, s_tok)].append(eid)

        print(f"  Indexed {c:,} records in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Total {total_s23:,} S2/S3 records indexed in {time.time()-t0:.1f}s.")
    print(f"  Exact name keys: {len(idx_exact):,}")
    print(f"  Core name keys:  {len(idx_core):,}")
    print(f"  Address keys:    {len(idx_addr):,}")

    # Step 2: Query Source 1 and write output files directly
    print("\nStep 2: Processing Source 1 queries and writing outputs...")
    t0 = time.time()
    total_s1 = 0
    s1_matched = 0
    total_cand_links = 0
    total_match_links = 0

    with open(s1_path, 'r', encoding='utf-8') as f_in, \
         open(matching_out, 'w', encoding='utf-8', newline='') as f_match, \
         open(candidate_out, 'w', encoding='utf-8', newline='') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.DictReader(f_in, delimiter='\t')
        for row in reader:
            total_s1 += 1
            s1_id = row['entity_id']
            country = fast_norm(row.get('country'))
            bname = row.get('business_name')
            addr = row.get('business_address') or ""

            norm_name = fast_norm(bname)
            core_name = extract_core_name(norm_name)
            b_num = extract_primary_number(addr)
            norm_addr = fast_norm(addr)
            s_tok = extract_first_street_token(norm_addr) if b_num > 0 else ""

            cands_s2 = []
            cands_s3 = []
            seen_cands = set()

            # 1. Exact Name Matches
            exact_list = idx_exact.get((country, norm_name), [])
            bucket_size = len(exact_list)
            for c_eid, c_num in exact_list:
                seen_cands.add(c_eid)
                if b_num > 0 and c_num > 0:
                    score = 100 if b_num == c_num else 0
                else:
                    score = 80 if bucket_size <= 3 else 0

                if score >= 80:
                    if c_eid.startswith('S2-'):
                        cands_s2.append((score, c_eid))
                    else:
                        cands_s3.append((score, c_eid))

            # 2. Core Name Matches (if exact had no matches)
            if not cands_s2 and not cands_s3 and core_name and core_name != norm_name:
                core_list = idx_core.get((country, core_name), [])
                core_bucket_size = len(core_list)
                for c_eid, c_num in core_list:
                    seen_cands.add(c_eid)
                    if b_num > 0 and c_num > 0:
                        score = 90 if b_num == c_num else 0
                    else:
                        score = 75 if core_bucket_size <= 2 else 0

                    if score >= 75:
                        if c_eid.startswith('S2-'):
                            cands_s2.append((score, c_eid))
                        else:
                            cands_s3.append((score, c_eid))

            # 3. Address Anchor Matches (recovers transliterated / acronym entities)
            if b_num > 0 and s_tok:
                addr_list = idx_addr.get((country, b_num, s_tok), [])
                if 0 < len(addr_list) <= 6:
                    for c_eid in addr_list:
                        seen_cands.add(c_eid)
                        if c_eid.startswith('S2-'):
                            cands_s2.append((70, c_eid))
                        else:
                            cands_s3.append((70, c_eid))

            # Sort descending by score and cap at top-2 per source
            cands_s2.sort(reverse=True)
            cands_s3.sort(reverse=True)

            top_matches = sorted(list(set([cid for sc, cid in cands_s2[:2]] + [cid for sc, cid in cands_s3[:2]])))
            all_candidates = sorted(list(seen_cands))

            match_str = ",".join(top_matches)
            cand_str = ",".join(all_candidates)

            if top_matches:
                s1_matched += 1
                total_match_links += len(top_matches)
            total_cand_links += len(all_candidates)

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            if total_s1 % 300000 == 0:
                print(f"  Processed {total_s1:,} queries in {time.time()-t0:.1f}s...")

    print(f"\nProcessing complete in {time.time()-t0:.1f}s.")
    print(f"Total pipeline runtime: {time.time()-t_total:.1f}s.")
    print("=" * 70)
    print("INFERENCE SUMMARY:")
    print(f"  Total S1 queries:        {total_s1:,}")
    print(f"  S1 queries with matches: {s1_matched:,} ({s1_matched/total_s1*100:.2f}%)")
    print(f"  Singletons (0 matches):  {total_s1 - s1_matched:,} ({(total_s1 - s1_matched)/total_s1*100:.2f}%)")
    print(f"  Total candidate links:   {total_cand_links:,} (avg {total_cand_links/total_s1:.2f} per S1)")
    print(f"  Total predicted matches: {total_match_links:,} (avg {total_match_links/total_s1:.2f} per S1)")
    print(f"  Outputs written to:")
    print(f"    {matching_out}")
    print(f"    {candidate_out}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Precision-First Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Test dataset directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()
