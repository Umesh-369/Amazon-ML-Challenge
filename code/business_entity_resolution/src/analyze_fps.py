#!/usr/bin/env python3
"""
Error Analysis: Dissecting the 144,725 False Positives in v03.
Pinpoints the exact rules causing false positives in v03 to enable surgical precision improvements.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, re, unicodedata, string, csv
from collections import defaultdict, Counter
import numpy as np

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

def main():
    train_dir = 'dataset/train'
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    val_indices_path = 'code/business_entity_resolution/val_indices.npy'

    print("=" * 70)
    print("ANALYZING v03 FALSE POSITIVES (FP BREAKDOWN)")
    print("=" * 70)

    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    val_indices = np.load(val_indices_path)
    sample_indices = set(val_indices[:50000])

    val_s1 = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in sample_indices:
                val_s1.append(row)

    print(f"Loaded {len(val_s1):,} validation sample queries.")

    # Build v03 indices
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_addr = defaultdict(list)

    for path in [s2_path, s3_path]:
        t0 = time.time()
        c = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                c += 1
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

                idx_exact[(country, norm_name)].append((eid, b_num))
                core_name = extract_core_name(norm_name)
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num))
                if b_num > 0:
                    s_tok = extract_first_street_token(norm_addr)
                    if s_tok:
                        idx_addr[(country, b_num, s_tok)].append(eid)

        print(f"  Indexed {c:,} in {os.path.basename(path)} in {time.time()-t0:.1f}s")

    # Evaluate FP causes
    fp_by_rule = Counter()
    total_fps = 0
    total_tps = 0

    for row in val_s1:
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

        # 1. Exact Name Matches
        exact_list = idx_exact.get((country, norm_name), [])
        bucket_size = len(exact_list)
        for c_eid, c_num in exact_list:
            if b_num > 0 and c_num > 0:
                score = 100 if b_num == c_num else 0
                rule = "exact_num_match"
            else:
                score = 80 if bucket_size <= 3 else 0
                rule = "exact_small_bucket_no_num"

            if score >= 80:
                if c_eid.startswith('S2-'): cands_s2.append((score, c_eid, rule))
                else: cands_s3.append((score, c_eid, rule))

        # 2. Core Name Matches
        if not cands_s2 and not cands_s3 and core_name and core_name != norm_name:
            core_list = idx_core.get((country, core_name), [])
            core_bucket_size = len(core_list)
            for c_eid, c_num in core_list:
                if b_num > 0 and c_num > 0:
                    score = 90 if b_num == c_num else 0
                    rule = "core_num_match"
                else:
                    score = 75 if core_bucket_size <= 2 else 0
                    rule = "core_small_bucket_no_num"

                if score >= 75:
                    if c_eid.startswith('S2-'): cands_s2.append((score, c_eid, rule))
                    else: cands_s3.append((score, c_eid, rule))

        # 3. Address Anchor Matches
        if b_num > 0 and s_tok:
            addr_list = idx_addr.get((country, b_num, s_tok), [])
            if 0 < len(addr_list) <= 6:
                for c_eid in addr_list:
                    rule = "addr_anchor_match"
                    if c_eid.startswith('S2-'): cands_s2.append((70, c_eid, rule))
                    else: cands_s3.append((70, c_eid, rule))

        cands_s2.sort(reverse=True, key=lambda x: x[0])
        cands_s3.sort(reverse=True, key=lambda x: x[0])

        top_pairs = [(cid, r) for sc, cid, r in cands_s2[:2]] + [(cid, r) for sc, cid, r in cands_s3[:2]]
        true_matches = gt_dict.get(s1_id, set())

        for cid, rule in top_pairs:
            if cid in true_matches:
                total_tps += 1
            else:
                total_fps += 1
                fp_by_rule[rule] += 1

    print("\n" + "=" * 70)
    print(f"FALSE POSITIVE BREAKDOWN ACROSS {total_fps:,} FALSE POSITIVES (TPs = {total_tps:,}):")
    print("=" * 70)
    for rule, count in fp_by_rule.most_common():
        print(f"  {rule:30s}: {count:6,d} ({count/total_fps*100:5.2f}%)")
    print("=" * 70)

if __name__ == '__main__':
    main()
