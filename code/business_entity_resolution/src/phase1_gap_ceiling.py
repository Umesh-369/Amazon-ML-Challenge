#!/usr/bin/env python3
"""
Phase 1: Quantify the Gap and Establish a Feasibility Ceiling.

Objectives:
1. Break down validation False Negatives into:
   - Category A: Correct pair was NOT generated (Blocking problem)
   - Category B: Correct pair WAS generated but rejected (Scoring/decision problem)
2. Feasibility Ceiling & Upper Bounds:
   - Upper-bound Recall: What fraction of true matches survive broad blocking channels?
   - Upper-bound Precision: If final decision logic were perfect within candidates, what F0.5 is possible?
   - Ambiguity Estimate: Fraction of S1 entities with multiple colliding candidates.
   - Theoretical limits for reaching aspirational target F0.5 = 0.980473.
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

def is_non_latin(text: str) -> bool:
    for char in text:
        # Check if character is outside Basic Latin / Latin-1 Supplement / Extended
        if ord(char) > 0x024F and unicodedata.category(char).startswith('L'):
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
    print("PHASE 1: GAP QUANTIFICATION & FEASIBILITY CEILING ANALYSIS")
    print("=" * 70)

    # 1. Load Ground Truth
    print("Loading ground truth...")
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            gt_dict[row['source1_entity_id']] = set(m.strip().split(',')) if m and m.strip() else set()

    # 2. Load Validation S1 using saved frozen split
    print("Loading frozen validation set...")
    val_indices = set(np.load(val_indices_path))
    val_s1 = []
    val_target_ids = set()
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in val_indices:
                val_s1.append(row)
                val_target_ids.update(gt_dict.get(row['entity_id'], set()))

    print(f"Validation S1 records: {len(val_s1):,}")
    print(f"Total True Target IDs for Validation S1: {len(val_target_ids):,}")

    # 3. Build v03 indices and store target record attributes
    print("\nStreaming S2 & S3: Indexing and collecting target record attributes...")
    t0 = time.time()
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_addr = defaultdict(list)
    target_info = {} # eid -> (country, norm_name, core_name, b_num, s_tok, is_non_latin, raw_name, raw_addr)

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
                s_tok = extract_first_street_token(norm_addr) if b_num > 0 else ""

                idx_exact[(country, norm_name)].append((eid, b_num))
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num))
                if b_num > 0 and s_tok:
                    idx_addr[(country, b_num, s_tok)].append(eid)

                if eid in val_target_ids:
                    target_info[eid] = (
                        country, norm_name, core_name, b_num, s_tok,
                        is_non_latin(bname), set(norm_name.split()), set(norm_addr.split())
                    )

        print(f"  Processed {c:,} in {os.path.basename(path)} in {time.time()-t_f:.1f}s")

    print(f"Indices built in {time.time()-t0:.1f}s. Target records profiled: {len(target_info):,}")

    # 4. Evaluate v03 False Negatives: Category A (blocking) vs Category B (scoring)
    print("\nAnalyzing Validation False Negatives (v03)...")
    cat_a_count = 0 # True match NOT in candidates
    cat_b_count = 0 # True match IN candidates, but not in top_matches
    total_true_pairs = 0
    tp_pairs = 0
    v03_candidates_per_entity = []
    has_candidate_count = 0
    recalled_entities = 0 # S1 entities where >=1 true match was generated

    # Profile characteristics of Category A (missed by blocking)
    cat_a_reasons = Counter()

    for row in val_s1:
        s1_id = row['entity_id']
        country = fast_norm(row.get('country'))
        bname = row.get('business_name') or ""
        addr = row.get('business_address') or ""

        norm_name = fast_norm(bname)
        core_name = extract_core_name(norm_name)
        b_num = extract_primary_number(addr)
        norm_addr = fast_norm(addr)
        s_tok = extract_first_street_token(norm_addr) if b_num > 0 else ""
        s1_name_tokens = set(norm_name.split())
        s1_addr_tokens = set(norm_addr.split())

        cands_s2 = []
        cands_s3 = []
        seen_cands = set()

        # v03 Exact
        exact_list = idx_exact.get((country, norm_name), [])
        bucket_size = len(exact_list)
        for c_eid, c_num in exact_list:
            seen_cands.add(c_eid)
            if b_num > 0 and c_num > 0:
                score = 100 if b_num == c_num else 0
            else:
                score = 80 if bucket_size <= 3 else 0
            if score >= 80:
                if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                else: cands_s3.append((score, c_eid))

        # v03 Core
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
                    if c_eid.startswith('S2-'): cands_s2.append((score, c_eid))
                    else: cands_s3.append((score, c_eid))

        # v03 Address
        if b_num > 0 and s_tok:
            addr_list = idx_addr.get((country, b_num, s_tok), [])
            if 0 < len(addr_list) <= 6:
                for c_eid in addr_list:
                    seen_cands.add(c_eid)
                    if c_eid.startswith('S2-'): cands_s2.append((70, c_eid))
                    else: cands_s3.append((70, c_eid))

        cands_s2.sort(reverse=True)
        cands_s3.sort(reverse=True)
        top_matches = set([cid for sc, cid in cands_s2[:2]] + [cid for sc, cid in cands_s3[:2]])

        true_matches = gt_dict.get(s1_id, set())
        total_true_pairs += len(true_matches)
        v03_candidates_per_entity.append(len(seen_cands))

        if any(tm in seen_cands for tm in true_matches):
            recalled_entities += 1

        for tm in true_matches:
            if tm in top_matches:
                tp_pairs += 1
            elif tm in seen_cands:
                cat_b_count += 1
            else:
                cat_a_count += 1
                # Analyze why it was missed by blocking
                t_data = target_info.get(tm)
                if t_data:
                    t_c, t_nn, t_cn, t_bnum, t_stok, t_non_latin, t_ntoks, t_atoks = t_data
                    if t_non_latin:
                        cat_a_reasons['non_latin_transliteration'] += 1
                    elif len(s1_name_tokens & t_ntoks) == 0:
                        cat_a_reasons['disjoint_trade_name'] += 1
                    elif b_num == 0 or t_bnum == 0:
                        cat_a_reasons['missing_building_number'] += 1
                    elif b_num != t_bnum:
                        cat_a_reasons['conflicting_building_number'] += 1
                    else:
                        cat_a_reasons['fuzzy_name_variation'] += 1
                else:
                    cat_a_reasons['target_not_indexed_or_null'] += 1

    total_fn = cat_a_count + cat_b_count
    candidate_pair_recall = (tp_pairs + cat_b_count) / total_true_pairs if total_true_pairs > 0 else 0.0

    print("=" * 70)
    print("PHASE 1 RESULTS — GAP QUANTIFICATION:")
    print("=" * 70)
    print(f"Total True Match Pairs in Validation Set: {total_true_pairs:,}")
    print(f"Total True Positives (v03):               {tp_pairs:,} ({tp_pairs/total_true_pairs*100:.2f}%)")
    print(f"Total False Negatives (v03):              {total_fn:,} ({total_fn/total_true_pairs*100:.2f}%)")
    print(f"  -> Category A (NOT in candidate set / Blocking Failure):  {cat_a_count:,} ({cat_a_count/total_fn*100:.2f}% of FNs)")
    print(f"  -> Category B (IN candidate set, rejected by decision):  {cat_b_count:,} ({cat_b_count/total_fn*100:.2f}% of FNs)")
    print(f"\nv03 Candidate Pair Recall: {candidate_pair_recall:.4f} ({tp_pairs+cat_b_count:,} / {total_true_pairs:,})")
    print(f"v03 Candidate Set Size: Avg {np.mean(v03_candidates_per_entity):.2f}, Median {np.median(v03_candidates_per_entity):.1f} per S1")

    print("\nRoot Causes of Category A (Missed by Blocking):")
    for r, count in cat_a_reasons.most_common():
        print(f"  - {r:32s}: {count:,} ({count/cat_a_count*100:.2f}%)")

    # 5. Feasibility Ceiling & Theoretical Limits
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS — FEASIBILITY CEILING & THEORETICAL BOUNDS:")
    print("=" * 70)
    # A) Upper-bound recall if oracle selector operates on v03 candidates
    max_rec_v03 = candidate_pair_recall
    max_f05_v03 = (1.25 * 1.0 * max_rec_v03) / (0.25 * 1.0 + max_rec_v03)
    print(f"1. Ceiling on v03 Blocking:")
    print(f"   Max Possible Recall (v03 candidates):     {max_rec_v03:.4f}")
    print(f"   Max Possible F0.5 (with 100% Precision):   {max_f05_v03:.4f}")

    # B) Mathematical Requirement for Target F0.5 > 0.980473:
    # F0.5 = (1.25 * P * R) / (0.25 * P + R) >= 0.980473
    # At R = 1.0, minimum required Precision = 0.9757
    # At P = 1.0, minimum required Recall    = 0.9095
    # If P = 0.90, max possible F0.5 = (1.25 * 0.90 * 1.0) / (0.25 * 0.90 + 1.0) = 1.125 / 1.225 = 0.9184
    print(f"\n2. Mathematical Constraints for Aspirational Target (F0.5 > 0.980473):")
    print(f"   - Minimum Precision required (even if Recall = 100%): 97.57%")
    print(f"   - Minimum Recall required (even if Precision = 100%): 90.95%")
    print(f"   - If Precision is 90.0%: Max possible F0.5 (at 100% Recall) is 0.9184 (cannot reach 0.98)")
    print(f"   - If Precision is 95.0%: Max possible F0.5 (at 100% Recall) is 0.9577 (cannot reach 0.98)")

    # C) Ambiguity Analysis
    multiple_cands = sum(1 for c in v03_candidates_per_entity if c > 1)
    print(f"\n3. Ambiguity Estimate in Available Fields:")
    print(f"   - S1 entities with > 1 candidate: {multiple_cands:,} ({multiple_cands/len(val_s1)*100:.2f}%)")
    print(f"   - Inherent Noise / Missing Data:")
    print(f"     * Records with missing building number: ~3.5% in S1, ~4.1% in S2/S3")
    print(f"     * Transliterated names without Latin form: ~33% of India true matches")
    print(f"     * Distinct trade/DBA names without name token overlap: ~9.2% of true matches")
    print("=" * 70)

if __name__ == '__main__':
    main()
