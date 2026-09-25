#!/usr/bin/env python3
"""
Phase 2: Blocking Channel Exploration & Recall Benchmarking.

Evaluates independent blocking channels and their combination on a representative
subset of the validation set (and full validation set) to measure:
1. True-match candidate recall per channel
2. Total candidates generated per channel (computational feasibility)
3. Incremental recall added to the union
4. Collision rate and candidate bucket distributions
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

def extract_all_numbers(addr: str):
    if not addr:
        return []
    nums = re.findall(r'\b\d+\b', addr)
    return [int(n) for n in nums if 0 < int(n) < 10000000][:3]

def extract_first_street_token(norm_addr: str) -> str:
    for t in norm_addr.split():
        if len(t) >= 4 and t not in STOP_WORDS and not t.isdigit():
            return t
    return ""

def extract_postcode(addr: str, country: str) -> str:
    # 6-digit pin code for India, 5-digit zip for US
    if not addr:
        return ""
    if country == 'india':
        m = re.search(r'\b[1-9]\d{5}\b', addr)
        return m.group(0) if m else ""
    elif country in ('united states', 'usa', 'us'):
        m = re.search(r'\b\d{5}\b', addr)
        return m.group(0) if m else ""
    elif country == 'france':
        m = re.search(r'\b\d{5}\b', addr)
        return m.group(0) if m else ""
    return ""

def main():
    train_dir = 'dataset/train'
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    val_indices_path = 'code/business_entity_resolution/val_indices.npy'

    print("=" * 70)
    print("PHASE 2: PROFILING CANDIDATE BLOCKING CHANNELS")
    print("=" * 70)

    # Load GT
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # Load Validation S1 sample (first 50,000 for high-speed channel evaluation)
    val_indices = np.load(val_indices_path)
    val_sample_indices = set(val_indices[:50000])

    val_sample_s1 = []
    val_sample_targets = set()
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in val_sample_indices:
                val_sample_s1.append(row)
                val_sample_targets.update(gt_dict.get(row['entity_id'], set()))

    print(f"Validation sample size: {len(val_sample_s1):,} S1 queries")
    print(f"Target matches in sample: {len(val_sample_targets):,}")

    # Inspect address sample to check postcode extraction
    print("\nSample Address Profiling (India & US):")
    sample_count = 0
    for row in val_sample_s1[:10]:
        c = fast_norm(row.get('country'))
        a = row.get('business_address') or ""
        p = extract_postcode(a, c)
        num = extract_primary_number(a)
        stok = extract_first_street_token(fast_norm(a))
        print(f"  [{c[:8]}] Addr: {a[:50]}... | Num: {num} | Street: {stok} | Postcode: {p}")

if __name__ == '__main__':
    main()
