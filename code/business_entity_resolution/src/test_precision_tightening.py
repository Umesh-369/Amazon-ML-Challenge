#!/usr/bin/env python3
"""
Precision Tightening & High-Recall Multi-Index Architecture.
Tests surgical refinement of address anchor + addition of sorted core and num_name channels.
Evaluated on 50,000 validation queries.
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
    print("TESTING REFINED MULTI-INDEX PIPELINE ON 50K VALIDATION SET")
    print("=" * 70)

    # 1. Load GT
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # 2. Sample 50k validation queries
    val_indices = np.load(val_indices_path)
    sample_indices = set(val_indices[:50000])

    val_s1 = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in sample_indices:
                val_s1.append(row)

    print(f"Loaded {len(val_s1):,} validation queries.")

    # 3. Build Indices (lightweight memory footprint)
    print("\nBuilding lightweight multi-channel indices...")
    t0 = time.time()
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_num_name = defaultdict(list)

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
                name_toks = extract_tokens(bname, min_len=4)

                # 1. Exact Name
                idx_exact[(country, norm_name)].append((eid, b_num))

                # 2. Core Name
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num))

                # 3. Sorted Core Name
                if sorted_core and sorted_core != core_name and len(idx_sorted_core[(country, sorted_core)]) < 20:
                    idx_sorted_core[(country, sorted_core)].append((eid, b_num))

                # 4. Address Anchor (b_num + street_tok)
                if b_num > 0 and s_tok and len(idx_addr[(country, b_num, s_tok)]) < 20:
                    idx_addr[(country, b_num, s_tok)].append(eid)

                # 5. Building Number + Primary Name Token
                if b_num > 0 and name_toks:
                    p_tok = name_toks[0]
                    if len(idx_num_name[(country, b_num, p_tok)]) < 15:
                        idx_num_name[(country, b_num, p_tok)].append((eid, s_tok))

        print(f"  Processed {c:,} in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indices built in {time.time()-t0:.1f}s.")

    # Test variations of address anchor max bucket and name token channel
    for max_addr_bkt in [1, 2, 3]:
        for use_num_name in [True, False]:
            precisions = []
            recalls = []
            correct_singletons = 0
            included_entities = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for row in val_s1:
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
                name_toks = extract_tokens(bname, min_len=4)

                cands_s2 = []
                cands_s3 = []

                # 1. Exact Name Matches
                exact_list = idx_exact.get((country, norm_name), [])
                bucket_size = len(exact_list)
                for c_eid, c_num in exact_list:
                    if b_num > 0 and c_num > 0:
                        score = 100 if b_num == c_num else 0
                    else:
                        score = 80 if bucket_size <= 3 else 0

                    if score >= 80:
                        if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                        else: cands_s3.append((score, c_eid))

                # 2. Core Name Matches (if exact had no match)
                if not cands_s2 and not cands_s3 and core_name and core_name != norm_name:
                    core_list = idx_core.get((country, core_name), [])
                    core_bucket_size = len(core_list)
                    for c_eid, c_num in core_list:
                        if b_num > 0 and c_num > 0:
                            score = 90 if b_num == c_num else 0
                        else:
                            score = 75 if core_bucket_size <= 2 else 0

                        if score >= 75:
                            if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                            else: cands_s3.append((score, c_eid))

                # 3. Sorted Core Name Matches (if no match yet)
                if not cands_s2 and not cands_s3 and sorted_core and sorted_core != core_name:
                    sc_list = idx_sorted_core.get((country, sorted_core), [])
                    sc_bkt = len(sc_list)
                    for c_eid, c_num in sc_list:
                        if b_num > 0 and c_num > 0:
                            score = 88 if b_num == c_num else 0
                        else:
                            score = 75 if sc_bkt <= 2 else 0

                        if score >= 75:
                            if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                            else: cands_s3.append((score, c_eid))

                # 4. Building Number + Primary Name Token
                if use_num_name and not cands_s2 and not cands_s3 and b_num > 0 and name_toks:
                    p_tok = name_toks[0]
                    nn_list = idx_num_name.get((country, b_num, p_tok), [])
                    nn_bkt = len(nn_list)
                    if 0 < nn_bkt <= 3:
                        for c_eid, c_stok in nn_list:
                            # If street tokens match or bucket is unique (1)
                            if s_tok and c_stok and s_tok == c_stok:
                                score = 85
                            elif nn_bkt == 1:
                                score = 75
                            else:
                                score = 0
                            if score >= 75:
                                if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                                else: cands_s3.append((score, c_eid))

                # 5. Address Anchor Matches (tightened)
                if not cands_s2 and not cands_s3 and b_num > 0 and s_tok:
                    addr_list = idx_addr.get((country, b_num, s_tok), [])
                    if 0 < len(addr_list) <= max_addr_bkt:
                        for c_eid in addr_list:
                            score = 72
                            if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                            else: cands_s3.append((score, c_eid))

                cands_s2.sort(reverse=True)
                cands_s3.sort(reverse=True)

                top_matches = set([cid for sc, cid in cands_s2[:2]] + [cid for sc, cid in cands_s3[:2]])
                true_matches = gt_dict.get(s1_id, set())

                tp = len(top_matches & true_matches)
                fp = len(top_matches - true_matches)
                fn = len(true_matches - top_matches)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                if len(true_matches) == 0 and len(top_matches) == 0:
                    correct_singletons += 1
                else:
                    included_entities += 1
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    precisions.append(p)
                    recalls.append(r)

            macro_p = float(np.mean(precisions))
            macro_r = float(np.mean(recalls))
            d = 0.25 * macro_p + macro_r
            f05 = (1.25 * macro_p * macro_r) / d if d > 0 else 0.0

            print(f"MaxAddrBkt={max_addr_bkt} | NumName={use_num_name!s:<5} | Macro P: {macro_p:.6f} | Macro R: {macro_r:.6f} | F0.5: {f05:.6f} | TP: {total_tp:6,d} | FP: {total_fp:6,d} | FN: {total_fn:6,d}")

if __name__ == '__main__':
    main()
