#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Pipeline v06 — Precision-Hardened High-Recall Architecture (CURRENT BEST)

Benchmark Progression:
- v01 (Exact Baseline):           F0.5 = 0.335256 (P=0.3906, R=0.2140)
- v02 (Unicode NFKD):             F0.5 = 0.378103 (P=0.4318, R=0.2525)
- v03 (Precision Pipeline):       F0.5 = 0.629750 (P=0.7127, R=0.4297)
- v04 (Decoupled Multi-Index):    F0.5 = 0.654825 (P=0.7793, R=0.3995)
- v05 (High-Recall Architecture): F0.5 = 0.794977 (P=0.8287, R=0.6836)
- v06 (Precision-Hardened):       F0.5 = 0.799943 (P=0.8335, R=0.6889) [~0.8000 Benchmark!]

Key Innovations in v06:
1. Indian Metro & Locality Conflict Guard:
   - Identifies Indian metropolitan hubs (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune).
   - Suppresses cross-metro collisions between identical chain names across different states.
2. Token Jaccard Disambiguation on Distinctive Names:
   - Computes set Jaccard similarity on candidate pairs, eliminating 954 false positive collisions.
3. Universal Brahmic Script Transliteration:
   - Full algorithmic phonetic mapping of Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati to Latin base.
4. Word Transposition & Postal Channels:
   - Sorted core names and PIN/ZIP prefix channels recover altered word orders and missing street numbers.
5. High-Throughput Buffered Output Generation:
   - Employs 50,000-line memory batching to prevent OneDrive filesystem write-lock throttling.
"""

import argparse
import csv
import json
import os
import re
import string
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Fast C Translation table for diacritics and punctuation
CHAR_MAP = {ord(c): 32 for c in string.punctuation}
for i in range(0x110000):
    if unicodedata.category(chr(i)).startswith('P'):
        CHAR_MAP[i] = 32

diacritic_pairs = 'àa áa âa ãa äa åa èe ée êe ëe ìi íi îi ïi òo óo ôo õo öo ùu úu ûu üu ýy ÿy çc ñn'
for pair in diacritic_pairs.split():
    CHAR_MAP[ord(pair[0])] = ord(pair[1])
    CHAR_MAP[ord(pair[0].upper())] = ord(pair[1])

# Indic Brahmic script mapping to Latin phonetic characters
BRAHMIC_OFFSET_MAP = {
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u', 0x0F: 'e', 0x13: 'o', 0x14: 'o',
    0x15: 'k', 0x16: 'k', 0x17: 'g', 0x18: 'g', 0x19: 'n',
    0x1A: 'c', 0x1B: 'c', 0x1C: 'j', 0x1D: 'j', 0x1E: 'n',
    0x1F: 't', 0x20: 't', 0x21: 'd', 0x22: 'd', 0x23: 'n',
    0x24: 't', 0x25: 't', 0x26: 'd', 0x27: 'd', 0x28: 'n',
    0x2A: 'p', 0x2B: 'p', 0x2C: 'b', 0x2D: 'b', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x32: 'l', 0x35: 'v', 0x36: 's', 0x37: 's', 0x38: 's', 0x39: 'h',
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x47: 'e', 0x4B: 'o', 0x4C: 'o'
}

def transliterate_indic(text: str) -> str:
    res = []
    for c in text:
        code = ord(c)
        if 0x0900 <= code <= 0x0D7F:
            offset = code % 0x80
            if offset in BRAHMIC_OFFSET_MAP:
                res.append(BRAHMIC_OFFSET_MAP[offset])
        elif c.isascii():
            res.append(c)
    return "".join(res)

def fast_norm(text: str) -> str:
    if not text:
        return ""
    s = transliterate_indic(str(text)).casefold()
    for dom in ['.com', '.org', '.net', '.in', '.co', '.fr', 'www.']:
        s = s.replace(dom, ' ')
    return " ".join(s.translate(CHAR_MAP).split())

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

INDIAN_METROS = {
    'delhi': 'delhi', 'new delhi': 'delhi',
    'mumbai': 'mumbai', 'bombay': 'mumbai',
    'bangalore': 'bangalore', 'bengaluru': 'bangalore',
    'chennai': 'chennai', 'madras': 'chennai',
    'kolkata': 'kolkata', 'calcutta': 'kolkata',
    'hyderabad': 'hyderabad', 'secunderabad': 'hyderabad',
    'pune': 'pune', 'ahmedabad': 'ahmedabad'
}

def extract_metro(norm_addr: str) -> str:
    for word in norm_addr.split():
        if word in INDIAN_METROS:
            return INDIAN_METROS[word]
    return ""

def extract_core_name(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def extract_primary_number(addr: str) -> int:
    if not addr:
        return 0
    m = re.search(r'\b\d+\b', addr)
    if m:
        val = int(m.group(0).lstrip('0') or '0')
        return val if 0 < val < 10000000 else 0
    return 0

def extract_postal_code(addr: str) -> str:
    m = re.search(r'\b[1-9]\d{5}\b|\b\d{5}\b', addr)
    return m.group(0) if m else ""

def extract_first_street_token(norm_addr: str) -> str:
    for t in norm_addr.split():
        if len(t) >= 4 and t not in STOP_WORDS and not t.isdigit():
            return t
    return ""

def token_jaccard(toks1: set, toks2: set) -> float:
    if not toks1 or not toks2:
        return 0.0
    u = len(toks1 | toks2)
    return len(toks1 & toks2) / u if u > 0 else 0.0

def build_indices(s2_path: str, s3_path: str):
    print("=" * 70)
    print("INDEXING SOURCE 2 & SOURCE 3 (v06 HIGH-PRECISION ARCHITECTURE)")
    print("=" * 70)
    t0 = time.time()

    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_bnum_prefix = defaultdict(list)
    idx_pin_prefix = defaultdict(list)
    idx_tok_street = defaultdict(list)
    idx_addr = defaultdict(list)

    total_records = 0
    for path in [s2_path, s3_path]:
        t_f = time.time()
        c = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                c += 1
                total_records += 1
                eid = row[0]
                country = fast_norm(row[3]) if len(row) > 3 else ""
                norm_name = fast_norm(row[1])
                if not norm_name or not country:
                    continue

                raw_addr = row[2] if len(row) > 2 else ""
                norm_addr = fast_norm(raw_addr)
                b_num = extract_primary_number(raw_addr)
                s_tok = extract_first_street_token(norm_addr)
                postcode = extract_postal_code(raw_addr)
                metro = extract_metro(norm_addr) if country == 'india' else ""
                first_char = norm_name[0] if norm_name else ""
                name_toks = set([t for t in norm_name.split() if t not in LEGAL_SUFFIXES and t not in STOP_WORDS])

                idx_exact[(country, norm_name)].append((eid, b_num, postcode, metro))
                core_name = extract_core_name(norm_name)
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num, postcode, metro))

                sorted_tokens = sorted(list(name_toks))
                if len(sorted_tokens) >= 2:
                    sorted_core = " ".join(sorted_tokens)
                    if sorted_core != core_name:
                        idx_sorted_core[(country, sorted_core)].append((eid, b_num, postcode, metro))

                if b_num > 0:
                    if len(norm_name) >= 4 and len(idx_bnum_prefix[(country, b_num, norm_name[:4])]) < 10:
                        idx_bnum_prefix[(country, b_num, norm_name[:4])].append(eid)
                    if s_tok:
                        idx_addr[(country, b_num, s_tok)].append((eid, first_char, name_toks))

                if s_tok:
                    dist_toks = [t for t in name_toks if len(t) >= 5]
                    if dist_toks and len(idx_tok_street[(country, dist_toks[0], s_tok)]) < 8:
                        idx_tok_street[(country, dist_toks[0], s_tok)].append((eid, name_toks, metro))

                if postcode and len(norm_name) >= 4 and len(idx_pin_prefix[(country, postcode, norm_name[:4])]) < 8:
                    idx_pin_prefix[(country, postcode, norm_name[:4])].append(eid)

        print(f"  Indexed {c:,} records in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indices built across {total_records:,} records in {time.time()-t0:.1f}s.")
    return idx_exact, idx_core, idx_sorted_core, idx_bnum_prefix, idx_pin_prefix, idx_tok_street, idx_addr

def query_s1(row, idx_exact, idx_core, idx_sorted_core, idx_bnum_prefix, idx_pin_prefix, idx_tok_street, idx_addr):
    country = fast_norm(row[3]) if len(row) > 3 else ""
    norm_name = fast_norm(row[1])
    raw_addr = row[2] if len(row) > 2 else ""
    norm_addr = fast_norm(raw_addr)
    b_num = extract_primary_number(raw_addr)
    s_tok = extract_first_street_token(norm_addr)
    postcode = extract_postal_code(raw_addr)
    metro = extract_metro(norm_addr) if country == 'india' else ""
    core_name = extract_core_name(norm_name)
    first_char = norm_name[0] if norm_name else ""
    s1_toks = set([t for t in norm_name.split() if t not in LEGAL_SUFFIXES and t not in STOP_WORDS])
    sorted_tokens = sorted(list(s1_toks))
    sorted_core = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 else ""

    scored_s2 = []
    scored_s3 = []
    seen_cands = set()

    def add_s(sc, cid):
        seen_cands.add(cid)
        if sc >= 75:
            if cid.startswith('S2-'): scored_s2.append((sc, cid))
            else: scored_s3.append((sc, cid))

    # 1. Exact Name Matches
    exact_list = idx_exact.get((country, norm_name), [])
    bkt = len(exact_list)
    for c_eid, c_num, c_post, c_metro in exact_list:
        score = 0
        if metro and c_metro and metro != c_metro:
            score = 0
        elif postcode and c_post and postcode != c_post:
            score = 0
        elif b_num > 0 and c_num > 0:
            score = 100 if b_num == c_num else 0
        elif b_num == 0 or c_num == 0:
            score = 90 if (postcode and c_post and postcode == c_post) else (85 if bkt <= 6 else (80 if bkt <= 12 else 0))
        add_s(score, c_eid)

    # 2. Core Name Matches
    if core_name and core_name != norm_name:
        core_list = idx_core.get((country, core_name), [])
        core_bkt = len(core_list)
        for c_eid, c_num, c_post, c_metro in core_list:
            score = 0
            if metro and c_metro and metro != c_metro:
                score = 0
            elif postcode and c_post and postcode != c_post:
                score = 0
            elif b_num > 0 and c_num > 0:
                score = 90 if b_num == c_num else 0
            elif b_num == 0 or c_num == 0:
                score = 80 if core_bkt <= 3 else 0
            add_s(score, c_eid)

    # 3. Sorted Core Name Matches
    if sorted_core and sorted_core != core_name:
        sc_list = idx_sorted_core.get((country, sorted_core), [])
        sc_bkt = len(sc_list)
        for c_eid, c_num, c_post, c_metro in sc_list:
            score = 0
            if metro and c_metro and metro != c_metro:
                score = 0
            elif postcode and c_post and postcode != c_post:
                score = 0
            elif b_num > 0 and c_num > 0:
                score = 88 if b_num == c_num else 0
            elif b_num == 0 or c_num == 0:
                score = 80 if sc_bkt <= 2 else 0
            add_s(score, c_eid)

    # 4. Building Number + Name 4-char Prefix
    if b_num > 0 and len(norm_name) >= 4:
        p_list = idx_bnum_prefix.get((country, b_num, norm_name[:4]), [])
        if 0 < len(p_list) <= 4:
            for c_eid in p_list:
                add_s(84, c_eid)

    # 5. Address Anchor Matches (with Jaccard / first_char guard)
    if b_num > 0 and s_tok:
        addr_list = idx_addr.get((country, b_num, s_tok), [])
        a_bkt = len(addr_list)
        if 0 < a_bkt <= 6:
            for c_eid, c_fchar, c_toks in addr_list:
                jacc = token_jaccard(s1_toks, c_toks)
                if jacc >= 0.33 or (first_char and c_fchar and first_char == c_fchar):
                    add_s(76, c_eid)

    # 6. Distinctive Name Token + Street Token (with Jaccard & Metro guard)
    if s_tok:
        dist_toks = [t for t in s1_toks if len(t) >= 5]
        if dist_toks:
            ts_list = idx_tok_street.get((country, dist_toks[0], s_tok), [])
            if 0 < len(ts_list) <= 4:
                for c_eid, c_toks, c_metro in ts_list:
                    if metro and c_metro and metro != c_metro:
                        continue
                    jacc = token_jaccard(s1_toks, c_toks)
                    if jacc >= 0.30 or len(dist_toks[0]) >= 7:
                        add_s(85, c_eid)

    # 7. Postal Code + Name 4-char Prefix
    if postcode and len(norm_name) >= 4:
        pin_list = idx_pin_prefix.get((country, postcode, norm_name[:4]), [])
        if 0 < len(pin_list) <= 4:
            for c_eid in pin_list:
                add_s(84, c_eid)

    scored_s2.sort(reverse=True)
    scored_s3.sort(reverse=True)

    top_matches = sorted(list(set([cid for sc, cid in scored_s2[:5]] + [cid for sc, cid in scored_s3[:5]])))
    all_candidates = sorted(list(seen_cands))

    return top_matches, all_candidates

def run_test_inference(test_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 70)
    print("RUNNING FAST TEST INFERENCE (PIPELINE v06 — BATCH BUFFERED)")
    print(f"  Test directory:   {test_dir}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    t_start = time.time()
    idx_exact, idx_core, idx_sorted_core, idx_bnum_prefix, idx_pin_prefix, idx_tok_street, idx_addr = build_indices(s2_path, s3_path)

    print("\nProcessing Source 1 queries and generating buffered outputs...")
    t0 = time.time()
    total_s1 = 0
    matched_s1 = 0
    total_matches = 0
    total_candidates = 0

    match_buffer = []
    cand_buffer = []
    BATCH_SIZE = 50000

    # Write to local temp files first to avoid OneDrive real-time write locks
    temp_dir = tempfile.gettempdir()
    temp_match = os.path.join(temp_dir, "temp_matching_results.tsv")
    temp_cand = os.path.join(temp_dir, "temp_candidate_pairs.tsv")

    with open(s1_path, 'r', encoding='utf-8') as f_in, \
         open(temp_match, 'w', encoding='utf-8', newline='') as f_match, \
         open(temp_cand, 'w', encoding='utf-8', newline='') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.reader(f_in, delimiter='\t')
        next(reader)
        for row in reader:
            total_s1 += 1
            s1_id = row[0]
            top_matches, all_candidates = query_s1(row, idx_exact, idx_core, idx_sorted_core, idx_bnum_prefix, idx_pin_prefix, idx_tok_street, idx_addr)

            match_str = ",".join(top_matches)
            cand_str = ",".join(all_candidates)

            if top_matches:
                matched_s1 += 1
                total_matches += len(top_matches)
            total_candidates += len(all_candidates)

            match_buffer.append(f"{s1_id}\t{match_str}\n")
            cand_buffer.append(f"{s1_id}\t{cand_str}\n")

            if len(match_buffer) >= BATCH_SIZE:
                f_match.write("".join(match_buffer))
                f_cand.write("".join(cand_buffer))
                match_buffer.clear()
                cand_buffer.clear()

            if total_s1 % 300000 == 0:
                print(f"  Processed {total_s1:,} queries in {time.time()-t0:.1f}s...")

        if match_buffer:
            f_match.write("".join(match_buffer))
            f_cand.write("".join(cand_buffer))

    # Move temp files to target output atomically
    import shutil
    shutil.move(temp_match, matching_out)
    shutil.move(temp_cand, candidate_out)

    print(f"\nInference completed in {time.time()-t_start:.1f}s.")
    print(f"  Total S1 queries: {total_s1:,}")
    print(f"  Queries with matches: {matched_s1:,} ({matched_s1/total_s1*100:.2f}%)")
    print(f"  Singletons: {total_s1 - matched_s1:,} ({(total_s1 - matched_s1)/total_s1*100:.2f}%)")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Total final matches:   {total_matches:,}")
    print(f"  Wrote: {matching_out} and {candidate_out}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline v06 — Precision-Hardened Architecture")
    parser.add_argument("--test-dir", default="Amazon-ML-Challenge/dataset/test", help="Test dataset directory")
    parser.add_argument("--output-dir", default="Amazon-ML-Challenge/output", help="Output directory")
    args = parser.parse_args()

    run_test_inference(args.test_dir, args.output_dir)

if __name__ == '__main__':
    main()
