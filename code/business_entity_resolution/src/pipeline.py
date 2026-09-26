#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Fast Streaming Submission Generator (v10 High-Ceiling Architecture)
Targeting 0.88 - 0.92+ Macro F0.5:
1. Inverted stream processing (indexes S1 ~1.73M entities into ~2.5GB RAM, streams S2 & S3).
2. Universal Indic Brahmic-to-Latin transliteration.
3. Fast Phonetic Soundex blocking (bridges Aggarwal/Agarwal, Chowdhury/Chaudhary, Prasad/Prashad).
4. Building number +/- 2 tolerance on identical street tokens.
5. Locality + Street + Soundex channel for unnumbered addresses (b_num == 0).
6. Character 3-gram fuzzy rescue for OCR/typo corruption.
7. Generic industry word suppression eliminating spurious brand collisions.
"""

import csv, os, sys, time, string, unicodedata, re, tempfile, shutil
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# Fast C Translation table
CHAR_MAP = {ord(c): 32 for c in string.punctuation}
for i in range(0x110000):
    if unicodedata.category(chr(i)).startswith('P'):
        CHAR_MAP[i] = 32

diacritic_pairs = 'àa áa âa ãa äa åa èe ée êe ëe ìi íi îi ïi òo óo ôo õo öo ùu úu ûu üu ýy ÿy çc ñn'
for pair in diacritic_pairs.split():
    CHAR_MAP[ord(pair[0])] = ord(pair[1])
    CHAR_MAP[ord(pair[0].upper())] = ord(pair[1])

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

GENERIC_INDUSTRY_WORDS = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'co', 'corp', 'corporation',
    'company', 'enterprises', 'enterprise', 'services', 'service', 'solutions', 'solution',
    'technologies', 'technology', 'group', 'holdings', 'llp', 'pllc', 'sa', 'sarl', 'sas',
    'construction', 'constructions', 'trading', 'builders', 'infra', 'infrastructure',
    'logistics', 'industries', 'industry', 'consultancy', 'consultants', 'consulting',
    'retail', 'agency', 'agencies', 'center', 'centre', 'motors', 'pharma', 'pharmaceuticals',
    'jewellers', 'jewellery', 'textiles', 'properties', 'realty', 'real estate', 'developers'
}

STOP_WORDS = {
    'the', 'and', 'of', 'in', 'at', 'on', 'near', 'opp', 'opposite', 'behind', 'floor',
    'road', 'street', 'st', 'rd', 'ave', 'avenue', 'lane', 'drive', 'dr', 'way', 'hwy', 'highway',
    'plot', 'block', 'sector', 'nagar', 'colony', 'apartment', 'apartments', 'bldg', 'building',
    'rue', 'avenue', 'boulevard', 'bd', 'chemin', 'place', 'flat', 'room', 'office', 'unit',
    'shop', 'cross', 'main', 'post', 'district', 'dist', 'taluk', 'tehsil', 'null'
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
    tokens = [t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS]
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

def fast_soundex(word: str) -> str:
    if not word:
        return ""
    w = word.lower()
    first = w[0]
    table = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    encoded = [first]
    for char in w[1:]:
        code = table.get(char, '')
        if code and code != encoded[-1]:
            encoded.append(code)
    res = "".join(encoded).replace(first, first, 1)
    return (res + "0000")[:4]

def get_char_3grams(text: str) -> set:
    s = text.replace(" ", "")
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s) - 2)}

def char_3gram_jaccard(g1: set, g2: set) -> float:
    if not g1 or not g2:
        return 0.0
    u = len(g1 | g2)
    return len(g1 & g2) / u if u > 0 else 0.0

def token_jaccard(toks1: set, toks2: set) -> float:
    if not toks1 or not toks2:
        return 0.0
    u = len(toks1 | toks2)
    return len(toks1 & toks2) / u if u > 0 else 0.0

def main():
    test_dir = 'Amazon-ML-Challenge/dataset/test'
    output_dir = 'Amazon-ML-Challenge/output'
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    print("=" * 75)
    print("PIPELINE v10 HIGH-CEILING STREAMING INFERENCE (TARGET: 0.88 - 0.92+)")
    print("=" * 75)
    t_start = time.time()

    # Step 1: Index Source 1 into compact structures (~2.0 GB RAM)
    print("Step 1/3: Ingesting & Indexing Source 1 (1.73M entities)...")
    t0 = time.time()

    s1_ids = []
    s1_bnum = []
    s1_post = []
    s1_metro = []
    s1_fchar = []
    s1_toks = []
    s1_3grams = []

    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_bnum_prefix = defaultdict(list)
    idx_pin_prefix = defaultdict(list)
    idx_tok_street = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_street_soundex = defaultdict(list)

    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for idx, row in enumerate(r):
            s1_id = row[0]
            s1_ids.append(s1_id)
            country = fast_norm(row[3]) if len(row) > 3 else ""
            norm_name = fast_norm(row[1])
            raw_addr = row[2] if len(row) > 2 else ""
            norm_addr = fast_norm(raw_addr)
            b_num = extract_primary_number(raw_addr)
            s_tok = extract_first_street_token(norm_addr)
            postcode = extract_postal_code(raw_addr)
            metro = extract_metro(norm_addr) if country == 'india' else ""
            first_char = norm_name[0] if norm_name else ""
            name_toks = set([t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
            name_3g = get_char_3grams(norm_name)
            core_name = extract_core_name(norm_name)
            sorted_tokens = sorted(list(name_toks))
            sorted_core = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 else ""

            s1_bnum.append(b_num)
            s1_post.append(postcode)
            s1_metro.append(metro)
            s1_fchar.append(first_char)
            s1_toks.append(name_toks)
            s1_3grams.append(name_3g)

            if norm_name and country:
                idx_exact[(country, norm_name)].append(idx)
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append(idx)
                if sorted_core and sorted_core != core_name:
                    idx_sorted_core[(country, sorted_core)].append(idx)
                if b_num > 0:
                    if len(norm_name) >= 4 and len(idx_bnum_prefix[(country, b_num, norm_name[:4])]) < 8:
                        idx_bnum_prefix[(country, b_num, norm_name[:4])].append(idx)
                    if s_tok:
                        idx_addr[(country, b_num, s_tok)].append(idx)
                if s_tok:
                    dist_toks = [t for t in name_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
                    if dist_toks and len(idx_tok_street[(country, dist_toks[0], s_tok)]) < 6:
                        idx_tok_street[(country, dist_toks[0], s_tok)].append(idx)
                    # Soundex channel for unnumbered addresses
                    if dist_toks:
                        sdx = fast_soundex(dist_toks[0])
                        if sdx and len(idx_street_soundex[(country, s_tok, metro if metro else "", sdx)]) < 5:
                            idx_street_soundex[(country, s_tok, metro if metro else "", sdx)].append(idx)
                if postcode and len(norm_name) >= 4 and len(idx_pin_prefix[(country, postcode, norm_name[:4])]) < 6:
                    idx_pin_prefix[(country, postcode, norm_name[:4])].append(idx)

            if (idx + 1) % 500000 == 0:
                print(f"  Indexed {idx+1:,} S1 entities in {time.time()-t0:.1f}s...")

    N = len(s1_ids)
    print(f"Source 1 fully indexed: {N:,} entities in {time.time()-t0:.1f}s.")

    s1_matches_s2 = [[] for _ in range(N)]
    s1_matches_s3 = [[] for _ in range(N)]
    s1_cands_s2 = [[] for _ in range(N)]
    s1_cands_s3 = [[] for _ in range(N)]

    # Step 2: Stream Source 2 (4.88M records)
    print("\nStep 2/3: Streaming Source 2 against v10 high-precision indices...")
    t0 = time.time()
    with open(s2_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            s2_id = row[0]
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
            name_toks = set([t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
            name_3g = get_char_3grams(norm_name)
            core_name = extract_core_name(norm_name)
            sorted_tokens = sorted(list(name_toks))
            sorted_core = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 else ""

            matched_s1_indices = {}

            # Exact Name
            for s1_i in idx_exact.get((country, norm_name), []):
                s1_m = s1_metro[s1_i]
                s1_p = s1_post[s1_i]
                s1_b = s1_bnum[s1_i]
                if metro and s1_m and metro != s1_m:
                    continue
                if postcode and s1_p and postcode != s1_p:
                    continue
                if b_num > 0 and s1_b > 0:
                    score = 100 if b_num == s1_b else (92 if abs(b_num - s1_b) <= 2 else 0)
                else:
                    score = 94 if (postcode and s1_p and postcode == s1_p) else 88
                if score >= 75:
                    matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Core Name
            if core_name and core_name != norm_name:
                for s1_i in idx_core.get((country, core_name), []):
                    s1_m = s1_metro[s1_i]
                    s1_p = s1_post[s1_i]
                    s1_b = s1_bnum[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    if postcode and s1_p and postcode != s1_p:
                        continue
                    if b_num > 0 and s1_b > 0:
                        score = 94 if b_num == s1_b else (88 if abs(b_num - s1_b) <= 2 else 0)
                    else:
                        score = 84
                    if score >= 75:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Sorted Core
            if sorted_core and sorted_core != core_name:
                for s1_i in idx_sorted_core.get((country, sorted_core), []):
                    s1_m = s1_metro[s1_i]
                    s1_p = s1_post[s1_i]
                    s1_b = s1_bnum[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    if postcode and s1_p and postcode != s1_p:
                        continue
                    if b_num > 0 and s1_b > 0:
                        score = 92 if b_num == s1_b else (86 if abs(b_num - s1_b) <= 2 else 0)
                    else:
                        score = 82
                    if score >= 75:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Bnum Prefix
            if b_num > 0 and len(norm_name) >= 4:
                for s1_i in idx_bnum_prefix.get((country, b_num, norm_name[:4]), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.25 or g_jacc >= 0.35:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 88)

            # Addr Anchor
            if b_num > 0 and s_tok:
                for s1_i in idx_addr.get((country, b_num, s_tok), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.25 or g_jacc >= 0.35 or (first_char and s1_fchar[s1_i] and first_char == s1_fchar[s1_i] and g_jacc >= 0.25):
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 82)

            # Dist Token + Street
            dist_toks = [t for t in name_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
            if s_tok and dist_toks:
                for s1_i in idx_tok_street.get((country, dist_toks[0], s_tok), []):
                    s1_m = s1_metro[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.30 or g_jacc >= 0.40:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 90)

                # Soundex channel for unnumbered addresses
                sdx = fast_soundex(dist_toks[0])
                if sdx:
                    for s1_i in idx_street_soundex.get((country, s_tok, metro if metro else "", sdx), []):
                        jacc = token_jaccard(name_toks, s1_toks[s1_i])
                        g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                        if jacc >= 0.33 or g_jacc >= 0.45:
                            matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 86)

            # PIN Prefix
            if postcode and len(norm_name) >= 4:
                for s1_i in idx_pin_prefix.get((country, postcode, norm_name[:4]), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.30 or g_jacc >= 0.40:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 86)

            for s1_i, sc in matched_s1_indices.items():
                if len(s1_cands_s2[s1_i]) < 8:
                    s1_cands_s2[s1_i].append(s2_id)
                if sc >= 75 and len(s1_matches_s2[s1_i]) < 6:
                    s1_matches_s2[s1_i].append((sc, s2_id))

            if (i + 1) % 1000000 == 0:
                print(f"  Streamed {i+1:,} Source 2 records in {time.time()-t0:.1f}s...")

    print(f"Source 2 streaming completed in {time.time()-t0:.1f}s.")

    # Step 3: Stream Source 3 (5.08M records)
    print("\nStep 3/3: Streaming Source 3 against v10 high-precision indices...")
    t0 = time.time()
    with open(s3_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            s3_id = row[0]
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
            name_toks = set([t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
            name_3g = get_char_3grams(norm_name)
            core_name = extract_core_name(norm_name)
            sorted_tokens = sorted(list(name_toks))
            sorted_core = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 else ""

            matched_s1_indices = {}

            # Exact Name
            for s1_i in idx_exact.get((country, norm_name), []):
                s1_m = s1_metro[s1_i]
                s1_p = s1_post[s1_i]
                s1_b = s1_bnum[s1_i]
                if metro and s1_m and metro != s1_m:
                    continue
                if postcode and s1_p and postcode != s1_p:
                    continue
                if b_num > 0 and s1_b > 0:
                    score = 100 if b_num == s1_b else (92 if abs(b_num - s1_b) <= 2 else 0)
                else:
                    score = 94 if (postcode and s1_p and postcode == s1_p) else 88
                if score >= 75:
                    matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Core Name
            if core_name and core_name != norm_name:
                for s1_i in idx_core.get((country, core_name), []):
                    s1_m = s1_metro[s1_i]
                    s1_p = s1_post[s1_i]
                    s1_b = s1_bnum[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    if postcode and s1_p and postcode != s1_p:
                        continue
                    if b_num > 0 and s1_b > 0:
                        score = 94 if b_num == s1_b else (88 if abs(b_num - s1_b) <= 2 else 0)
                    else:
                        score = 84
                    if score >= 75:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Sorted Core
            if sorted_core and sorted_core != core_name:
                for s1_i in idx_sorted_core.get((country, sorted_core), []):
                    s1_m = s1_metro[s1_i]
                    s1_p = s1_post[s1_i]
                    s1_b = s1_bnum[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    if postcode and s1_p and postcode != s1_p:
                        continue
                    if b_num > 0 and s1_b > 0:
                        score = 92 if b_num == s1_b else (86 if abs(b_num - s1_b) <= 2 else 0)
                    else:
                        score = 82
                    if score >= 75:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), score)

            # Bnum Prefix
            if b_num > 0 and len(norm_name) >= 4:
                for s1_i in idx_bnum_prefix.get((country, b_num, norm_name[:4]), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.25 or g_jacc >= 0.35:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 88)

            # Addr Anchor
            if b_num > 0 and s_tok:
                for s1_i in idx_addr.get((country, b_num, s_tok), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.25 or g_jacc >= 0.35 or (first_char and s1_fchar[s1_i] and first_char == s1_fchar[s1_i] and g_jacc >= 0.25):
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 82)

            # Dist Token + Street
            dist_toks = [t for t in name_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
            if s_tok and dist_toks:
                for s1_i in idx_tok_street.get((country, dist_toks[0], s_tok), []):
                    s1_m = s1_metro[s1_i]
                    if metro and s1_m and metro != s1_m:
                        continue
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.30 or g_jacc >= 0.40:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 90)

                # Soundex channel for unnumbered addresses
                sdx = fast_soundex(dist_toks[0])
                if sdx:
                    for s1_i in idx_street_soundex.get((country, s_tok, metro if metro else "", sdx), []):
                        jacc = token_jaccard(name_toks, s1_toks[s1_i])
                        g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                        if jacc >= 0.33 or g_jacc >= 0.45:
                            matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 86)

            # PIN Prefix
            if postcode and len(norm_name) >= 4:
                for s1_i in idx_pin_prefix.get((country, postcode, norm_name[:4]), []):
                    jacc = token_jaccard(name_toks, s1_toks[s1_i])
                    g_jacc = char_3gram_jaccard(name_3g, s1_3grams[s1_i])
                    if jacc >= 0.30 or g_jacc >= 0.40:
                        matched_s1_indices[s1_i] = max(matched_s1_indices.get(s1_i, 0), 86)

            for s1_i, sc in matched_s1_indices.items():
                if len(s1_cands_s3[s1_i]) < 8:
                    s1_cands_s3[s1_i].append(s3_id)
                if sc >= 75 and len(s1_matches_s3[s1_i]) < 6:
                    s1_matches_s3[s1_i].append((sc, s3_id))

            if (i + 1) % 1000000 == 0:
                print(f"  Streamed {i+1:,} Source 3 records in {time.time()-t0:.1f}s...")

    print(f"Source 3 streaming completed in {time.time()-t0:.1f}s.")

    # Step 4: Write Final Submission Files
    print("\nStep 4: Writing final submission files...")
    t0 = time.time()

    temp_dir = tempfile.gettempdir()
    temp_match = os.path.join(temp_dir, "temp_stream_matching_v10.tsv")
    temp_cand = os.path.join(temp_dir, "temp_stream_candidate_v10.tsv")

    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    matched_count = 0
    total_matches = 0

    with open(temp_match, 'w', encoding='utf-8', newline='') as f_m, \
         open(temp_cand, 'w', encoding='utf-8', newline='') as f_c:
        
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        m_buf = []
        c_buf = []

        for i in range(N):
            s1_id = s1_ids[i]

            s1_matches_s2[i].sort(reverse=True)
            s1_matches_s3[i].sort(reverse=True)

            top_matches = [cid for sc, cid in s1_matches_s2[i][:4]] + [cid for sc, cid in s1_matches_s3[i][:4]]
            all_cands = s1_cands_s2[i][:8] + s1_cands_s3[i][:8]

            match_str = ",".join(top_matches)
            cand_str = ",".join(all_cands)

            if top_matches:
                matched_count += 1
                total_matches += len(top_matches)

            m_buf.append(f"{s1_id}\t{match_str}\n")
            c_buf.append(f"{s1_id}\t{cand_str}\n")

            if len(m_buf) >= 50000:
                f_m.write("".join(m_buf))
                f_c.write("".join(c_buf))
                m_buf.clear()
                c_buf.clear()

        if m_buf:
            f_m.write("".join(m_buf))
            f_c.write("".join(c_buf))

    shutil.move(temp_match, matching_out)
    shutil.move(temp_cand, candidate_out)

    print(f"\nFiles generated in {time.time()-t0:.1f}s!")
    print(f"  Total S1 rows written: {N:,} (100% of test entities)")
    print(f"  S1 entities with matches: {matched_count:,} ({matched_count/N*100:.2f}%)")
    print(f"  Singletons: {N - matched_count:,} ({(N - matched_count)/N*100:.2f}%)")
    print(f"  Total execution time: {time.time()-t_start:.1f}s")
    print(f"  Destination 1: {matching_out}")
    print(f"  Destination 2: {candidate_out}")

if __name__ == '__main__':
    main()
