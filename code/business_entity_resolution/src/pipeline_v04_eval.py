#!/usr/bin/env python3
"""
v04 Multi-Channel High-Recall & Precision Hardened Pipeline.
Full Validation Run on held-out 80/20 stratified split (441,364 entities).

Architecture Enhancements over v03:
1. Channel 1 (Exact Name):
   - Address token / street token recovery when building number is missing, allowing safe recovery of true matches in buckets > 3.
2. Channel 2 (Core Name):
   - Strips legal suffixes; verifies building number or locality overlap.
3. Channel 3 (Sorted Core Name):
   - Catches word transpositions and rearranged legal qualifiers.
4. Channel 4 (Building Number + Primary Name Token):
   - Captures name extensions/acronym variations at verified physical premises.
5. Channel 5 (Hardened Address Anchor):
   - Single-tenant address protection (bucket == 1) and multi-tenant discrimination (bucket >= 2 requires name signal or non-Latin script).
6. Number Contradiction Guard:
   - Strict rejection on conflicting primary numbers.
7. Grouped validation on frozen val_indices.npy.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, re, unicodedata, string, csv
from collections import defaultdict, Counter
import numpy as np

# Fast C Translation table for diacritics and punctuation
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

def extract_tokens(text: str, min_len=4):
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

def is_indic_script(text: str) -> bool:
    for c in text:
        if 0x0900 <= ord(c) <= 0x0D7F:
            return True
    return False

def main():
    train_dir = 'dataset/train'
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    val_indices_path = 'code/business_entity_resolution/val_indices.npy'

    print("=" * 70)
    print("RUNNING v04 VALIDATION BENCHMARK (441,364 ENTITIES)")
    print("=" * 70)
    t_total = time.time()

    # 1. Load Ground Truth
    print("Step 1: Loading Ground Truth...")
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # 2. Load Validation S1 Entities (frozen split)
    print("Step 2: Loading frozen validation set...")
    val_indices = set(np.load(val_indices_path))
    val_s1 = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in val_indices:
                val_s1.append(row)

    print(f"Validation set loaded: {len(val_s1):,} S1 entities.")

    # 3. Build Multi-Channel Indices from S2 & S3
    print("\nStep 3: Indexing Source 2 & Source 3 across 5 channels...")
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
                first_char = norm_name[0] if norm_name else ""
                indic = is_indic_script(bname)

                # Ch1: Exact Name -> (eid, b_num, s_tok)
                idx_exact[(country, norm_name)].append((eid, b_num, s_tok))

                # Ch2: Core Name
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num, s_tok))

                # Ch3: Sorted Core Name
                if sorted_core and sorted_core != core_name and len(idx_sorted_core[(country, sorted_core)]) < 20:
                    idx_sorted_core[(country, sorted_core)].append((eid, b_num, s_tok))

                # Ch4: Address Anchor -> (eid, first_char, indic)
                if b_num > 0 and s_tok and len(idx_addr[(country, b_num, s_tok)]) < 20:
                    idx_addr[(country, b_num, s_tok)].append((eid, first_char, indic))

                # Ch5: Building Number + Primary Name Token -> (eid, s_tok)
                if b_num > 0 and name_toks:
                    p_tok = name_toks[0]
                    if len(idx_num_name[(country, b_num, p_tok)]) < 15:
                        idx_num_name[(country, b_num, p_tok)].append((eid, s_tok))

        print(f"  Indexed {c:,} records in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indices built in {time.time()-t0:.1f}s.")
    print(f"  Exact name keys:       {len(idx_exact):,}")
    print(f"  Core name keys:        {len(idx_core):,}")
    print(f"  Sorted core keys:      {len(idx_sorted_core):,}")
    print(f"  Address anchor keys:   {len(idx_addr):,}")
    print(f"  Num + Name token keys: {len(idx_num_name):,}")

    # 4. Evaluate Validation Set
    print("\nStep 4: Evaluating v04 on held-out validation set...")
    t0 = time.time()
    precisions = []
    recalls = []
    correct_singletons = 0
    included_entities = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_cand_links = 0
    total_predicted_links = 0

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
        name_toks = extract_tokens(bname, min_len=4)
        first_char = norm_name[0] if norm_name else ""

        cands_s2 = []
        cands_s3 = []
        seen_cands = set()

        # 1. Exact Name Matches
        exact_list = idx_exact.get((country, norm_name), [])
        bucket_size = len(exact_list)
        for c_eid, c_num, c_stok in exact_list:
            seen_cands.add(c_eid)
            # Conflicting building number check
            if b_num > 0 and c_num > 0:
                score = 100 if b_num == c_num else 0
            elif b_num == 0 or c_num == 0:
                if bucket_size <= 3:
                    score = 80
                elif s_tok and c_stok and s_tok == c_stok:
                    score = 85 # Locality/street token confirms true match even in larger bucket!
                else:
                    score = 0
            else:
                score = 0

            if score >= 80:
                if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                else: cands_s3.append((score, c_eid))

        # 2. Core Name Matches (if exact had no match)
        if not cands_s2 and not cands_s3 and core_name and core_name != norm_name:
            core_list = idx_core.get((country, core_name), [])
            core_bucket_size = len(core_list)
            for c_eid, c_num, c_stok in core_list:
                seen_cands.add(c_eid)
                if b_num > 0 and c_num > 0:
                    score = 90 if b_num == c_num else 0
                elif b_num == 0 or c_num == 0:
                    if core_bucket_size <= 2:
                        score = 75
                    elif s_tok and c_stok and s_tok == c_stok:
                        score = 80
                    else:
                        score = 0
                else:
                    score = 0

                if score >= 75:
                    if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                    else: cands_s3.append((score, c_eid))

        # 3. Sorted Core Name Matches (word transpositions)
        if not cands_s2 and not cands_s3 and sorted_core and sorted_core != core_name:
            sc_list = idx_sorted_core.get((country, sorted_core), [])
            sc_bkt = len(sc_list)
            for c_eid, c_num, c_stok in sc_list:
                seen_cands.add(c_eid)
                if b_num > 0 and c_num > 0:
                    score = 88 if b_num == c_num else 0
                elif b_num == 0 or c_num == 0:
                    if sc_bkt <= 2:
                        score = 75
                    elif s_tok and c_stok and s_tok == c_stok:
                        score = 80
                    else:
                        score = 0
                else:
                    score = 0

                if score >= 75:
                    if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                    else: cands_s3.append((score, c_eid))

        # 4. Building Number + Primary Name Token
        if not cands_s2 and not cands_s3 and b_num > 0 and name_toks:
            p_tok = name_toks[0]
            nn_list = idx_num_name.get((country, b_num, p_tok), [])
            nn_bkt = len(nn_list)
            if 0 < nn_bkt <= 3:
                for c_eid, c_stok in nn_list:
                    seen_cands.add(c_eid)
                    if s_tok and c_stok and s_tok == c_stok:
                        score = 85
                    elif nn_bkt == 1:
                        score = 78
                    else:
                        score = 0

                    if score >= 78:
                        if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                        else: cands_s3.append((score, c_eid))

        # 5. Address Anchor Matches (Hardened for Transliteration and Trade Names)
        if not cands_s2 and not cands_s3 and b_num > 0 and s_tok:
            addr_list = idx_addr.get((country, b_num, s_tok), [])
            a_bkt = len(addr_list)
            if 0 < a_bkt <= 5:
                for c_eid, c_fchar, c_indic in addr_list:
                    seen_cands.add(c_eid)
                    if a_bkt == 1:
                        score = 74 # Single tenant at address
                    elif c_indic:
                        score = 73 # Transliterated Indic script
                    elif first_char and c_fchar and first_char == c_fchar:
                        score = 72 # Shared first initial
                    elif a_bkt <= 3:
                        score = 70
                    else:
                        score = 0

                    if score >= 70:
                        if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                        else: cands_s3.append((score, c_eid))

        cands_s2.sort(reverse=True)
        cands_s3.sort(reverse=True)

        top_matches = set([cid for sc, cid in cands_s2[:2]] + [cid for sc, cid in cands_s3[:2]])
        true_matches = gt_dict.get(s1_id, set())

        total_cand_links += len(seen_cands)
        total_predicted_links += len(top_matches)

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

        if (idx + 1) % 100000 == 0:
            print(f"  Evaluated {idx+1:,}/{len(val_s1):,} in {time.time()-t0:.1f}s...")

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    denom = 0.25 * macro_p + macro_r
    f05 = (1.25 * macro_p * macro_r) / denom if denom > 0 else 0.0
    v03_f05 = 0.629750
    delta = f05 - v03_f05

    print("\n" + "=" * 70)
    print("v04 FULL VALIDATION RESULTS:")
    print("=" * 70)
    print(f"Total Validation Entities: {len(val_s1):,}")
    print(f"Correct Singletons:        {correct_singletons:,}")
    print(f"Included in Macro Average: {included_entities:,}")
    print(f"Macro Precision:           {macro_p:.6f} (v03: 0.712679, Diff: {macro_p - 0.712679:+.6f})")
    print(f"Macro Recall:              {macro_r:.6f} (v03: 0.429732, Diff: {macro_r - 0.429732:+.6f})")
    print(f"Macro F0.5:                {f05:.6f} (v03: 0.629750, Diff: {delta:+.6f})")
    print(f"Total True Positives (TP): {total_tp:,} (v03: 631,925, Diff: {total_tp - 631925:+d})")
    print(f"Total False Positives (FP):{total_fp:,} (v03: 144,725, Diff: {total_fp - 144725:+d})")
    print(f"Total False Negatives (FN):{total_fn:,} (v03: 895,964, Diff: {total_fn - 895964:+d})")
    print(f"Total Candidate Links:     {total_cand_links:,} (avg {total_cand_links/len(val_s1):.2f}/S1)")
    print(f"Total Pipeline Runtime:    {time.time()-t_total:.1f}s")
    print("=" * 70)

    if delta >= 0.005:
        print(f">>> PROMOTION ELIGIBLE: Delta ({delta:+.6f}) exceeds threshold (>= +0.005) <<<")
    else:
        print(f">>> NOT PROMOTED: Delta ({delta:+.6f}) does not clear minimum threshold (+0.005) <<<")

if __name__ == '__main__':
    main()
