#!/usr/bin/env python3
"""
Deep Ground-Truth Analysis on True Match Pairs.
Measures the exact coverage (% of true matches captured) for various potential blocking keys:
1. Exact normalized name
2. Core name (legal suffixes stripped)
3. Sorted core name tokens
4. Address anchor: (building_number, street_token)
5. Building number + Name token
6. Building number + Name 4-char prefix
7. Distinct name token + Distinct address token
8. First 2 name tokens
9. Indian script transliteration via Indic-Latin / Soundex / unidecode / script mapping
10. Trade name mismatch patterns
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

def soundex(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'[^a-z]', '', name.lower())
    if not name:
        return ""
    first = name[0].upper()
    mapping = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    encoded = [first]
    prev = mapping.get(name[0], '')
    for char in name[1:]:
        code = mapping.get(char, '')
        if code:
            if code != prev:
                encoded.append(code)
            prev = code
        else:
            prev = ''
    return "".join(encoded[:4]).ljust(4, '0')

def is_indic_script(text: str) -> bool:
    for c in text:
        cp = ord(c)
        # Devanagari: 0900-097F, Bengali: 0980-09FF, Gurmukhi: 0A00-0A7F, Gujarati: 0A80-0AFF,
        # Oriya: 0B00-0B7F, Tamil: 0B80-0BFF, Telugu: 0C00-0C7F, Kannada: 0C80-0CFF, Malayalam: 0D00-0D7F
        if 0x0900 <= cp <= 0x0D7F:
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
    print("ANALYZING TRUE MATCH PAIRS TO DESIGN OPTIMAL BLOCKING CHANNELS")
    print("=" * 70)

    # 1. Load GT
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            m = row['matched_entity_ids']
            if m and m.strip():
                gt_dict[row['source1_entity_id']] = set(m.strip().split(','))

    # 2. Select 25,000 validation S1 entities with true matches
    val_indices = np.load(val_indices_path)
    sample_indices = set(val_indices[:25000])

    s1_dict = {}
    target_to_s1 = defaultdict(list)
    total_pairs = 0

    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(r):
            if idx in sample_indices:
                eid = row['entity_id']
                s1_dict[eid] = row
                for tid in gt_dict.get(eid, set()):
                    target_to_s1[tid].append(eid)
                    total_pairs += 1

    print(f"Sample S1 queries: {len(s1_dict):,}, Total true pairs to analyze: {total_pairs:,}")

    # 3. Stream S2 & S3 to fetch target records
    t0 = time.time()
    target_records = {}
    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f, delimiter='\t')
            for row in r:
                eid = row['entity_id']
                if eid in target_to_s1:
                    target_records[eid] = row
                    if len(target_records) == len(target_to_s1):
                        break
        if len(target_records) == len(target_to_s1):
            break

    print(f"Collected all {len(target_records):,} target records in {time.time()-t0:.1f}s")

    # 4. Measure Channel Coverage across True Pairs
    channel_hits = Counter()
    union_hits = set()
    v03_hits = set()

    pair_idx = 0
    indic_pairs = 0
    trade_name_pairs = 0
    no_num_pairs = 0

    for tid, s1_ids in target_to_s1.items():
        t_row = target_records.get(tid)
        if not t_row:
            continue
        t_c = fast_norm(t_row.get('country'))
        t_name = t_row.get('business_name') or ""
        t_addr = t_row.get('business_address') or ""
        t_norm_name = fast_norm(t_name)
        t_core_name = extract_core_name(t_norm_name)
        t_sorted_core = extract_sorted_core_name(t_norm_name)
        t_num = extract_primary_number(t_addr)
        t_norm_addr = fast_norm(t_addr)
        t_stok = extract_first_street_token(t_norm_addr)
        t_name_tokens = set(extract_tokens(t_name, min_len=3))
        t_addr_tokens = set(extract_tokens(t_addr, min_len=3))
        t_first_tok = t_norm_name.split()[0] if t_norm_name else ""
        t_soundex = soundex(t_first_tok)
        t_is_indic = is_indic_script(t_name)

        for sid in s1_ids:
            pair_idx += 1
            pair_id = (sid, tid)
            s_row = s1_dict[sid]
            s_c = fast_norm(s_row.get('country'))
            s_name = s_row.get('business_name') or ""
            s_addr = s_row.get('business_address') or ""
            s_norm_name = fast_norm(s_name)
            s_core_name = extract_core_name(s_norm_name)
            s_sorted_core = extract_sorted_core_name(s_norm_name)
            s_num = extract_primary_number(s_addr)
            s_norm_addr = fast_norm(s_addr)
            s_stok = extract_first_street_token(s_norm_addr)
            s_name_tokens = set(extract_tokens(s_name, min_len=3))
            s_addr_tokens = set(extract_tokens(s_addr, min_len=3))
            s_first_tok = s_norm_name.split()[0] if s_norm_name else ""
            s_soundex = soundex(s_first_tok)

            if t_is_indic:
                indic_pairs += 1
            if len(s_name_tokens & t_name_tokens) == 0:
                trade_name_pairs += 1
            if s_num == 0 or t_num == 0:
                no_num_pairs += 1

            # Test Channels:
            # 1. Exact Name
            h_exact = (s_c == t_c and s_norm_name == t_norm_name and s_norm_name != "")
            if h_exact: channel_hits['ch1_exact_name'] += 1

            # 2. Core Name
            h_core = (s_c == t_c and s_core_name == t_core_name and s_core_name != "")
            if h_core: channel_hits['ch2_core_name'] += 1

            # 3. Sorted Core Name
            h_sorted_core = (s_c == t_c and s_sorted_core == t_sorted_core and s_sorted_core != "")
            if h_sorted_core: channel_hits['ch3_sorted_core_name'] += 1

            # 4. Address Anchor (b_num + street_tok)
            h_addr_anchor = (s_c == t_c and s_num > 0 and s_num == t_num and s_stok and s_stok == t_stok)
            if h_addr_anchor: channel_hits['ch4_addr_anchor'] += 1

            # 5. Building Number + Name Token
            h_num_name_tok = (s_c == t_c and s_num > 0 and s_num == t_num and len(s_name_tokens & t_name_tokens) > 0)
            if h_num_name_tok: channel_hits['ch5_num_name_token'] += 1

            # 6. Building Number + Name 4-char prefix
            s_pref = s_norm_name[:4] if len(s_norm_name) >= 4 else ""
            t_pref = t_norm_name[:4] if len(t_norm_name) >= 4 else ""
            h_num_pref = (s_c == t_c and s_num > 0 and s_num == t_num and s_pref and s_pref == t_pref)
            if h_num_pref: channel_hits['ch6_num_name_4prefix'] += 1

            # 7. Name Token + Address Token
            h_name_addr_tok = (s_c == t_c and len(s_name_tokens & t_name_tokens) > 0 and len(s_addr_tokens & t_addr_tokens) > 0)
            if h_name_addr_tok: channel_hits['ch7_name_tok_addr_tok'] += 1

            # 8. First 2 Name Tokens match
            s_toks = s_norm_name.split()[:2]
            t_toks = t_norm_name.split()[:2]
            h_first_2_toks = (s_c == t_c and len(s_toks) >= 2 and s_toks == t_toks)
            if h_first_2_toks: channel_hits['ch8_first_2_name_tokens'] += 1

            # 9. Soundex first token + Address number
            h_soundex_num = (s_c == t_c and s_soundex and s_soundex == t_soundex and s_num > 0 and s_num == t_num)
            if h_soundex_num: channel_hits['ch9_soundex_bnum'] += 1

            # 10. Address building number + any common address token (for transliterations/trade names)
            h_bnum_addrtok = (s_c == t_c and s_num > 0 and s_num == t_num and len(s_addr_tokens & t_addr_tokens) >= 1)
            if h_bnum_addrtok: channel_hits['ch10_bnum_addrtok'] += 1

            # Check v03 coverage on this pair
            h_v03 = (h_exact or h_core or h_addr_anchor)
            if h_v03: v03_hits.add(pair_id)

            # Check Union of Proposed Extended Channels
            h_union = (h_exact or h_core or h_sorted_core or h_addr_anchor or h_num_name_tok or h_bnum_addrtok or h_first_2_toks)
            if h_union: union_hits.add(pair_id)

    print("\n" + "=" * 70)
    print(f"EVALUATION OF BLOCKING CHANNELS ACROSS {pair_idx:,} TRUE MATCH PAIRS:")
    print("=" * 70)
    for ch, count in sorted(channel_hits.items()):
        rec = count / pair_idx
        print(f"  {ch:26s}: {count:6,} / {pair_idx:,} ({rec*100:5.2f}%)")

    v03_rec = len(v03_hits) / pair_idx
    union_rec = len(union_hits) / pair_idx
    print("-" * 70)
    print(f"Current v03 Blocking Recall: {len(v03_hits):,} / {pair_idx:,} ({v03_rec*100:.2f}%)")
    print(f"Proposed Union Recall:      {len(union_hits):,} / {pair_idx:,} ({union_rec*100:.2f}%)")
    print(f"Relative Recall Gain:       +{(union_rec - v03_rec)*100:.2f}% (from {v03_rec*100:.1f}% to {union_rec*100:.1f}%)")
    print("=" * 70)

if __name__ == '__main__':
    main()
