#!/usr/bin/env python3
"""
Phase 2: Computational Feasibility Guard & Multi-Channel Candidate Sizing.

Measures:
1. Candidate count per S1 query across each channel.
2. Candidate posting list size distribution and capping behavior.
3. True match candidate recall on 50,000 validation queries.
4. Total candidates in union to ensure memory and runtime stay within constraints.
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

def extract_sorted_core_name(norm_name: str) -> str:
    tokens = sorted([t for t in norm_name.split() if t not in LEGAL_SUFFIXES and t not in STOP_WORDS])
    return " ".join(tokens)

def extract_tokens(text: str, min_len=3):
    norm_s = fast_norm(text)
    return [t for t in norm_s.split() if len(t) >= min_len and t not in STOP_WORDS and t not in LEGAL_SUFFIXES]

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
    print("PHASE 2: COMPUTATIONAL FEASIBILITY & CANDIDATE SET SIZING")
    print("=" * 70)

    # 1. Load GT
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # 2. Select 50,000 validation queries
    val_indices = np.load(val_indices_path)
    sample_indices = set(val_indices[:50000])

    val_s1 = []
    total_true_matches = 0
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in sample_indices:
                val_s1.append(row)
                total_true_matches += len(gt_dict.get(row['entity_id'], set()))

    print(f"Loaded {len(val_s1):,} validation queries with {total_true_matches:,} true match pairs.")

    # 3. Build Multi-Channel Inverted Indices with Cap Guard
    # Caps per bucket to prevent any single key blowing up candidate count:
    MAX_EXACT_BUCKET = 50
    MAX_CORE_BUCKET = 30
    MAX_ADDR_BUCKET = 20
    MAX_NUM_NAME_BUCKET = 20
    MAX_FIRST2_BUCKET = 20

    print("\nBuilding Inverted Indices from train_source2.tsv & train_source3.tsv...")
    t0 = time.time()
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_num_name = defaultdict(list)
    idx_first2 = defaultdict(list)

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
                bname = row.get('business_name') or ""
                addr = row.get('business_address') or ""

                norm_name = fast_norm(bname)
                if not norm_name or not country:
                    continue

                b_num = extract_primary_number(addr)
                norm_addr = fast_norm(addr)
                core_name = extract_core_name(norm_name)
                sorted_core = extract_sorted_core_name(norm_name)
                s_tok = extract_first_street_token(norm_addr) if b_num > 0 else ""
                name_toks = extract_tokens(bname, min_len=3)
                words = norm_name.split()

                # Ch1: Exact Name
                if len(idx_exact[(country, norm_name)]) < MAX_EXACT_BUCKET:
                    idx_exact[(country, norm_name)].append(eid)

                # Ch2: Core Name
                if core_name and core_name != norm_name:
                    if len(idx_core[(country, core_name)]) < MAX_CORE_BUCKET:
                        idx_core[(country, core_name)].append(eid)

                # Ch3: Sorted Core Name (tokens sorted)
                if sorted_core and sorted_core != core_name:
                    if len(idx_sorted_core[(country, sorted_core)]) < MAX_CORE_BUCKET:
                        idx_sorted_core[(country, sorted_core)].append(eid)

                # Ch4: Address Anchor (b_num + street_tok)
                if b_num > 0 and s_tok:
                    if len(idx_addr[(country, b_num, s_tok)]) < MAX_ADDR_BUCKET:
                        idx_addr[(country, b_num, s_tok)].append(eid)

                # Ch5: Building Number + Name Token
                if b_num > 0 and name_toks:
                    for t in name_toks[:2]:
                        if len(idx_num_name[(country, b_num, t)]) < MAX_NUM_NAME_BUCKET:
                            idx_num_name[(country, b_num, t)].append(eid)

                # Ch6: First 2 Name Tokens
                if len(words) >= 2:
                    k_f2 = (country, words[0], words[1])
                    if len(idx_first2[k_f2]) < MAX_FIRST2_BUCKET:
                        idx_first2[k_f2].append(eid)

        print(f"  Processed {c:,} in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indexing complete in {time.time()-t0:.1f}s.")
    print(f"  Exact name keys:       {len(idx_exact):,}")
    print(f"  Core name keys:        {len(idx_core):,}")
    print(f"  Sorted core keys:      {len(idx_sorted_core):,}")
    print(f"  Address anchor keys:   {len(idx_addr):,}")
    print(f"  Num + Name token keys: {len(idx_num_name):,}")
    print(f"  First 2 token keys:    {len(idx_first2):,}")

    # 4. Measure Candidates & True Recall per Channel and in Union
    print("\nEvaluating Candidate Generation across 50,000 queries...")
    t0 = time.time()
    ch_cands = Counter()
    ch_hits = Counter()
    union_hits = 0
    total_union_cands = 0
    cands_per_query = []

    for idx, row in enumerate(val_s1):
        s1_id = row['entity_id']
        country = fast_norm(row.get('country'))
        bname = row.get('business_name') or ""
        addr = row.get('business_address') or ""

        norm_name = fast_norm(bname)
        core_name = extract_core_name(norm_name)
        sorted_core = extract_sorted_core_name(norm_name)
        b_num = extract_primary_number(addr)
        norm_addr = fast_norm(addr)
        s_tok = extract_first_street_token(norm_addr) if b_num > 0 else ""
        name_toks = extract_tokens(bname, min_len=3)
        words = norm_name.split()

        true_matches = gt_dict.get(s1_id, set())

        # Collect per channel
        cands_ch1 = set(idx_exact.get((country, norm_name), []))
        cands_ch2 = set(idx_core.get((country, core_name), [])) if core_name else set()
        cands_ch3 = set(idx_sorted_core.get((country, sorted_core), [])) if sorted_core else set()
        cands_ch4 = set(idx_addr.get((country, b_num, s_tok), [])) if (b_num > 0 and s_tok) else set()
        cands_ch5 = set()
        if b_num > 0 and name_toks:
            for t in name_toks[:2]:
                cands_ch5.update(idx_num_name.get((country, b_num, t), []))
        cands_ch6 = set(idx_first2.get((country, words[0], words[1]), [])) if len(words) >= 2 else set()

        all_union = cands_ch1 | cands_ch2 | cands_ch3 | cands_ch4 | cands_ch5 | cands_ch6
        total_union_cands += len(all_union)
        cands_per_query.append(len(all_union))

        # Check true hits
        for tm in true_matches:
            if tm in cands_ch1: ch_hits['ch1_exact'] += 1
            if tm in cands_ch2: ch_hits['ch2_core'] += 1
            if tm in cands_ch3: ch_hits['ch3_sorted_core'] += 1
            if tm in cands_ch4: ch_hits['ch4_addr_anchor'] += 1
            if tm in cands_ch5: ch_hits['ch5_num_name'] += 1
            if tm in cands_ch6: ch_hits['ch6_first2'] += 1
            if tm in all_union: union_hits += 1

        ch_cands['ch1_exact'] += len(cands_ch1)
        ch_cands['ch2_core'] += len(cands_ch2)
        ch_cands['ch3_sorted_core'] += len(cands_ch3)
        ch_cands['ch4_addr_anchor'] += len(cands_ch4)
        ch_cands['ch5_num_name'] += len(cands_ch5)
        ch_cands['ch6_first2'] += len(cands_ch6)

    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS: CHANNEL-BY-CHANNEL CANDIDATE SIZING & RECALL:")
    print("=" * 70)
    print(f"{'Channel':<24s} | {'Candidate Count':<16s} | {'Avg/Query':<10s} | {'True Hits':<12s} | {'Recall':<8s}")
    print("-" * 76)
    for ch in ['ch1_exact', 'ch2_core', 'ch3_sorted_core', 'ch4_addr_anchor', 'ch5_num_name', 'ch6_first2']:
        c_cnt = ch_cands[ch]
        c_avg = c_cnt / len(val_s1)
        h_cnt = ch_hits[ch]
        rec = h_cnt / total_true_matches if total_true_matches > 0 else 0.0
        print(f"{ch:<24s} | {c_cnt:<16,d} | {c_avg:<10.2f} | {h_cnt:<12,d} | {rec*100:6.2f}%")

    print("-" * 76)
    union_recall = union_hits / total_true_matches
    print(f"CUMULATIVE UNION: Total Candidates = {total_union_cands:,} (avg {total_union_cands/len(val_s1):.2f}/query)")
    print(f"Median Candidates per Query: {np.median(cands_per_query):.1f}, 95th Percentile: {np.percentile(cands_per_query, 95):.1f}")
    print(f"CUMULATIVE UNION TRUE MATCH RECALL: {union_hits:,} / {total_true_matches:,} ({union_recall*100:.2f}%)")
    print(f"(Baseline v03 Candidate Recall was 55.18% — Union achieves {union_recall*100:.2f}%!)")
    print("=" * 70)

if __name__ == '__main__':
    main()
