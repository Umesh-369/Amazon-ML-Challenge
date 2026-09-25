#!/usr/bin/env python3
"""
Reproduction Check for v03 Pipeline.
Runs the exact fixed 80/20 stratified validation methodology on train_source1.tsv
and verifies that v03 reproduces:
    Precision = 0.712679
    Recall    = 0.429732
    F0.5      = 0.629750
    FP        = 144,725
    FN        = 895,964
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, re, unicodedata, string, csv
from collections import defaultdict
import numpy as np

# Fast C translation table
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

    print("=" * 70)
    print("STEP 0: REPRODUCTION CHECK FOR v03 PIPELINE")
    print("=" * 70)

    # 1. Load Ground Truth
    print("Loading ground truth...")
    t0 = time.time()
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            gt_dict[row['source1_entity_id']] = set(m.strip().split(',')) if m and m.strip() else set()
    print(f"Loaded ground truth for {len(gt_dict):,} S1 entities in {time.time()-t0:.1f}s")

    # 2. Load Train S1 & compute fixed stratified split
    print("Loading train_source1.tsv and creating 80/20 stratified split (seed=42)...")
    t0 = time.time()
    s1_records = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            s1_records.append(row)
    print(f"Loaded {len(s1_records):,} S1 records in {time.time()-t0:.1f}s")

    has_match = np.array([len(gt_dict.get(row['entity_id'], set())) > 0 for row in s1_records], dtype=bool)
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

    print(f"Split sizes: Train={len(train_indices):,}, Val={len(val_indices):,}")

    # Save frozen split indices so all experiments use the exact same array
    split_dir = 'code/business_entity_resolution'
    np.save(os.path.join(split_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(split_dir, 'train_indices.npy'), train_indices)
    print("Saved val_indices.npy and train_indices.npy for reuse across all phases.")

    val_s1 = [s1_records[i] for i in val_indices]

    # 3. Build v03 Inverted Indices from S2 and S3
    print("\nBuilding v03 indices from train_source2.tsv and train_source3.tsv...")
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

    print(f"Indexed total {total_s23:,} S2/S3 records in {time.time()-t0:.1f}s.")
    print(f"  Exact name keys: {len(idx_exact):,}")
    print(f"  Core name keys:  {len(idx_core):,}")
    print(f"  Address keys:    {len(idx_addr):,}")

    # 4. Evaluate on Held-out Validation Set
    print("\nEvaluating v03 on held-out validation set (441,364 entities)...")
    t0 = time.time()
    precisions = []
    recalls = []
    correct_singletons = 0
    included_entities = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_cands = 0

    for idx, row in enumerate(val_s1):
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

        # 3. Address Anchor Matches
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

        top_matches = set([cid for sc, cid in cands_s2[:2]] + [cid for sc, cid in cands_s3[:2]])
        total_cands += len(seen_cands)

        true_matches = gt_dict.get(s1_id, set())

        tp = len(top_matches.intersection(true_matches))
        fp = len(top_matches - true_matches)
        fn = len(true_matches - top_matches)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(true_matches) == 0 and len(top_matches) == 0:
            correct_singletons += 1
        else:
            included_entities += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        if (idx + 1) % 100000 == 0:
            print(f"  Evaluated {idx+1:,}/{len(val_s1):,} in {time.time()-t0:.1f}s...")

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    denom = 0.25 * macro_prec + macro_rec
    f05 = (1.25 * macro_prec * macro_rec) / denom if denom > 0 else 0.0

    print("\n" + "=" * 70)
    print("v03 REPRODUCTION CHECK RESULTS:")
    print("=" * 70)
    print(f"Total Validation Entities: {len(val_s1):,}")
    print(f"Correct Singletons:        {correct_singletons:,}")
    print(f"Included in Macro Average: {included_entities:,}")
    print(f"Macro Precision:           {macro_prec:.6f} (Target v03: 0.712679, Diff: {macro_prec - 0.712679:+.6f})")
    print(f"Macro Recall:              {macro_rec:.6f} (Target v03: 0.429732, Diff: {macro_rec - 0.429732:+.6f})")
    print(f"Macro F0.5:                {f05:.6f} (Target v03: 0.629750, Diff: {f05 - 0.629750:+.6f})")
    print(f"Total TP:                  {total_tp:,}")
    print(f"Total FP:                  {total_fp:,} (Target v03: 144,725, Diff: {total_fp - 144725:+d})")
    print(f"Total FN:                  {total_fn:,} (Target v03: 895,964, Diff: {total_fn - 895964:+d})")
    print("=" * 70)

    if abs(macro_prec - 0.712679) < 0.0001 and abs(macro_rec - 0.429732) < 0.0001 and abs(f05 - 0.629750) < 0.0001:
        print(">>> REPRODUCTION CHECK PASSED! Exact match with documented benchmark. <<<")
    else:
        print(">>> WARNING: Reproduction check discrepancy detected! <<<")

if __name__ == '__main__':
    main()
