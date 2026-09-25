#!/usr/bin/env python3
"""
v04 Prototype Scorer: Multi-Channel Blocking + Structured Address Engine.
Tests scoring weights on a 50k validation sample to optimize Macro F0.5.
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

def score_pair(s_name, s_core, s_num, s_stok, s_toks, s_atok_set,
               c_name, c_core, c_num, c_stok, c_toks, c_atok_set, bucket_size):
    """
    Precision-first scoring function:
    Combines name agreement, address agreement, contradiction detection, and bucket penalties.
    """
    # 1. Contradiction Detection: Conflicting building numbers
    if s_num > 0 and c_num > 0 and s_num != c_num:
        # Strict contradiction penalty
        return 0.0

    num_match = (s_num > 0 and s_num == c_num)
    stok_match = (s_stok != "" and s_stok == c_stok)
    addr_tok_overlap = len(s_atok_set & c_atok_set)

    # 2. Name Matching Levels
    if s_name == c_name and s_name:
        if num_match:
            return 100.0
        elif s_num == 0 or c_num == 0:
            if bucket_size <= 2:
                return 92.0
            elif bucket_size <= 5:
                return 85.0
            elif addr_tok_overlap >= 1 or stok_match:
                return 88.0
            else:
                return 0.0 # Common name, no address confirmation -> suppress to avoid chain FP

    if s_core and c_core and s_core == c_core:
        if num_match:
            return 95.0
        elif s_num == 0 or c_num == 0:
            if bucket_size <= 2:
                return 85.0
            elif addr_tok_overlap >= 1 or stok_match:
                return 82.0
            else:
                return 0.0

    # Token overlap for names
    common_name_toks = len(s_toks & c_toks)
    if common_name_toks >= 2:
        if num_match:
            return 90.0
        elif (s_num == 0 or c_num == 0) and (stok_match or addr_tok_overlap >= 2):
            return 80.0

    if common_name_toks >= 1:
        if num_match and (stok_match or addr_tok_overlap >= 1):
            return 85.0

    # Address Anchor Match (recovers transliterations and trade names)
    if num_match and stok_match:
        if bucket_size <= 4:
            return 78.0

    return 0.0

def main():
    train_dir = 'dataset/train'
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    val_indices_path = 'code/business_entity_resolution/val_indices.npy'

    print("=" * 70)
    print("v04 TUNING: Multi-Channel Blocking + Structured Address Engine")
    print("=" * 70)

    # Load GT
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # Sample 50k validation queries
    val_indices = np.load(val_indices_path)
    sample_indices = set(val_indices[:50000])

    val_s1 = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in sample_indices:
                val_s1.append(row)

    print(f"Loaded {len(val_s1):,} validation queries.")

    # Build Indices
    t0 = time.time()
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_num_name = defaultdict(list)
    idx_first2 = defaultdict(list)
    s23_data = {} # eid -> (norm_name, core_name, b_num, s_tok, name_tok_set, addr_tok_set)

    for path in [s2_path, s3_path]:
        t_f = time.time()
        c = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                c += 1
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
                addr_toks = extract_tokens(addr, min_len=3)
                words = norm_name.split()

                s23_data[eid] = (norm_name, core_name, b_num, s_tok, set(name_toks), set(addr_toks))

                if len(idx_exact[(country, norm_name)]) < 50:
                    idx_exact[(country, norm_name)].append(eid)
                if core_name and core_name != norm_name and len(idx_core[(country, core_name)]) < 30:
                    idx_core[(country, core_name)].append(eid)
                if sorted_core and sorted_core != core_name and len(idx_sorted_core[(country, sorted_core)]) < 30:
                    idx_sorted_core[(country, sorted_core)].append(eid)
                if b_num > 0 and s_tok and len(idx_addr[(country, b_num, s_tok)]) < 20:
                    idx_addr[(country, b_num, s_tok)].append(eid)
                if b_num > 0 and name_toks:
                    for t in name_toks[:2]:
                        if len(idx_num_name[(country, b_num, t)]) < 20:
                            idx_num_name[(country, b_num, t)].append(eid)
                if len(words) >= 2 and len(idx_first2[(country, words[0], words[1])]) < 20:
                    idx_first2[(country, words[0], words[1])].append(eid)

        print(f"  Processed {c:,} in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indices built in {time.time()-t0:.1f}s.")

    # Test several operating thresholds
    for min_score in [75.0, 78.0, 80.0, 82.0, 85.0]:
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
            name_toks = set(extract_tokens(bname, min_len=3))
            addr_toks = set(extract_tokens(addr, min_len=3))
            words = norm_name.split()

            # Gather candidates
            cands = set(idx_exact.get((country, norm_name), []))
            if core_name: cands.update(idx_core.get((country, core_name), []))
            if sorted_core: cands.update(idx_sorted_core.get((country, sorted_core), []))
            if b_num > 0 and s_tok: cands.update(idx_addr.get((country, b_num, s_tok), []))
            if b_num > 0 and name_toks:
                for t in list(name_toks)[:2]:
                    cands.update(idx_num_name.get((country, b_num, t), []))
            if len(words) >= 2:
                cands.update(idx_first2.get((country, words[0], words[1]), []))

            scored_s2 = []
            scored_s3 = []
            for cid in cands:
                c_data = s23_data.get(cid)
                if not c_data:
                    continue
                c_norm_name, c_core, c_bnum, c_stok, c_name_toks, c_addr_toks = c_data
                sc = score_pair(norm_name, core_name, b_num, s_tok, name_toks, addr_toks,
                                c_norm_name, c_core, c_bnum, c_stok, c_name_toks, c_addr_toks, len(cands))
                if sc >= min_score:
                    if cid.startswith('S2-'): scored_s2.append((sc, cid))
                    else: scored_s3.append((sc, cid))

            scored_s2.sort(reverse=True)
            scored_s3.sort(reverse=True)

            top_matches = set([cid for sc, cid in scored_s2[:2]] + [cid for sc, cid in scored_s3[:2]])
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
        print(f"Threshold = {min_score:4.1f} | Macro P: {macro_p:.6f} | Macro R: {macro_r:.6f} | F0.5: {f05:.6f} | TP: {total_tp:6,d} | FP: {total_fp:6,d} | FN: {total_fn:6,d}")

if __name__ == '__main__':
    main()
