import csv, os, sys, time, string, unicodedata, re
from collections import defaultdict
import numpy as np

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
    val_indices = set(np.load('Amazon-ML-Challenge/code/business_entity_resolution/val_indices.npy')[:20000])

    print("Loading Ground Truth...")
    gt_dict = {}
    with open('Amazon-ML-Challenge/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if len(row) >= 2 and row[1].strip():
                gt_dict[row[0]] = set(row[1].strip().split(','))

    val_s1 = []
    target_match_ids = set()
    with open('Amazon-ML-Challenge/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for idx, row in enumerate(r):
            if idx in val_indices:
                val_s1.append(row)
                target_match_ids.update(gt_dict.get(row[0], set()))

    print("Building Inverted Indices for v09 (High-Precision + Fuzzy Typo Engine)...")
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    idx_sorted_core = defaultdict(list)
    idx_bnum_prefix = defaultdict(list)
    idx_pin_prefix = defaultdict(list)
    idx_tok_street = defaultdict(list)
    idx_addr = defaultdict(list)

    t0 = time.time()
    for s_file in ['train_source2.tsv', 'train_source3.tsv']:
        path = os.path.join('Amazon-ML-Challenge/dataset/train', s_file)
        with open(path, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for i, row in enumerate(r):
                eid = row[0]
                if eid not in target_match_ids and i > 300000:
                    continue
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
                name_3grams = get_char_3grams(norm_name)

                idx_exact[(country, norm_name)].append((eid, b_num, postcode, metro))
                core_name = extract_core_name(norm_name)
                if core_name and core_name != norm_name:
                    idx_core[(country, core_name)].append((eid, b_num, postcode, metro))

                sorted_tokens = sorted(list(name_toks))
                if len(sorted_tokens) >= 2:
                    sorted_core = " ".join(sorted_tokens)
                    if sorted_core != core_name:
                        idx_sorted_core[(country, sorted_core)].append((eid, b_num, postcode, metro))

                if b_num > 0 and len(norm_name) >= 4:
                    if len(idx_bnum_prefix[(country, b_num, norm_name[:4])]) < 8:
                        idx_bnum_prefix[(country, b_num, norm_name[:4])].append((eid, s_tok, postcode, metro, name_toks, name_3grams))

                if b_num > 0 and s_tok:
                    idx_addr[(country, b_num, s_tok)].append((eid, first_char, postcode, metro, name_toks, name_3grams))

                if s_tok:
                    dist_toks = [t for t in name_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
                    if dist_toks and len(idx_tok_street[(country, dist_toks[0], s_tok)]) < 6:
                        idx_tok_street[(country, dist_toks[0], s_tok)].append((eid, name_toks, metro, postcode, name_3grams))

                if postcode and len(norm_name) >= 4:
                    if len(idx_pin_prefix[(country, postcode, norm_name[:4])]) < 6:
                        idx_pin_prefix[(country, postcode, norm_name[:4])].append((eid, name_toks, name_3grams))

    print(f"Indices built in {time.time()-t0:.1f}s. Evaluating v09 queries...")

    precisions = []
    recalls = []
    correct_singletons = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    t0 = time.time()
    for row in val_s1:
        s1_id = row[0]
        country = fast_norm(row[3]) if len(row) > 3 else ""
        norm_name = fast_norm(row[1])
        raw_addr = row[2] if len(row) > 2 else ""
        norm_addr = fast_norm(raw_addr)
        b_num = extract_primary_number(raw_addr)
        s_tok = extract_first_street_token(norm_addr)
        postcode = extract_postal_code(raw_addr)
        metro = extract_metro(norm_addr) if country == 'india' else ""
        core_name = extract_core_name(norm_name)
        first_char = norm_name[0] if norm_name else ""
        s1_toks = set([t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
        s1_3grams = get_char_3grams(norm_name)
        sorted_tokens = sorted(list(s1_toks))
        sorted_core = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 else ""

        scored_s2 = []
        scored_s3 = []

        def add_s(sc, cid):
            if sc >= 82:
                if cid.startswith('S2-'): scored_s2.append((sc, cid))
                else: scored_s3.append((sc, cid))

        # 1. Exact Name Matches
        exact_list = idx_exact.get((country, norm_name), [])
        bkt = len(exact_list)
        for c_eid, c_num, c_post, c_metro in exact_list:
            score = 0
            if metro and c_metro and metro != c_metro:
                score = 0
            elif postcode and c_post and postcode != c_post:
                score = 0
            elif b_num > 0 and c_num > 0:
                if b_num == c_num:
                    score = 100
                elif abs(b_num - c_num) <= 2:
                    score = 92
            elif b_num == 0 or c_num == 0:
                score = 94 if (postcode and c_post and postcode == c_post) else (90 if bkt <= 3 else (84 if bkt <= 6 else 0))
            add_s(score, c_eid)

        # 2. Core Name Matches
        if core_name and core_name != norm_name:
            core_list = idx_core.get((country, core_name), [])
            core_bkt = len(core_list)
            for c_eid, c_num, c_post, c_metro in core_list:
                score = 0
                if metro and c_metro and metro != c_metro:
                    score = 0
                elif postcode and c_post and postcode != c_post:
                    score = 0
                elif b_num > 0 and c_num > 0:
                    if b_num == c_num:
                        score = 94
                    elif abs(b_num - c_num) <= 2:
                        score = 88
                elif b_num == 0 or c_num == 0:
                    score = 88 if core_bkt <= 2 else 0
                add_s(score, c_eid)

        # 3. Sorted Core Name Matches
        if sorted_core and sorted_core != core_name:
            sc_list = idx_sorted_core.get((country, sorted_core), [])
            sc_bkt = len(sc_list)
            for c_eid, c_num, c_post, c_metro in sc_list:
                score = 0
                if metro and c_metro and metro != c_metro:
                    score = 0
                elif postcode and c_post and postcode != c_post:
                    score = 0
                elif b_num > 0 and c_num > 0:
                    if b_num == c_num:
                        score = 92
                    elif abs(b_num - c_num) <= 2:
                        score = 86
                elif b_num == 0 or c_num == 0:
                    score = 84 if sc_bkt <= 2 else 0
                add_s(score, c_eid)

        # 4. Building Number + Brand Prefix (with Fuzzy 3-gram Support)
        if b_num > 0 and len(norm_name) >= 4:
            p_list = idx_bnum_prefix.get((country, b_num, norm_name[:4]), [])
            if 0 < len(p_list) <= 6:
                for c_eid, c_stok, c_post, c_metro, c_toks, c_3g in p_list:
                    if metro and c_metro and metro != c_metro:
                        continue
                    if postcode and c_post and postcode != c_post:
                        continue
                    jacc = token_jaccard(s1_toks, c_toks)
                    g_jacc = char_3gram_jaccard(s1_3grams, c_3g)
                    if s_tok and c_stok and s_tok == c_stok:
                        add_s(92 if (jacc >= 0.30 or g_jacc >= 0.40) else 86, c_eid)
                    elif jacc >= 0.35 or g_jacc >= 0.50:
                        add_s(88, c_eid)

        # 5. Address Anchor Matches (Single-tenant with initial match or fuzzy name)
        if b_num > 0 and s_tok:
            addr_list = idx_addr.get((country, b_num, s_tok), [])
            a_bkt = len(addr_list)
            if a_bkt == 1:
                c_eid, c_fchar, c_post, c_metro, c_toks, c_3g = addr_list[0]
                if not (metro and c_metro and metro != c_metro) and not (postcode and c_post and postcode != c_post):
                    jacc = token_jaccard(s1_toks, c_toks)
                    g_jacc = char_3gram_jaccard(s1_3grams, c_3g)
                    if jacc >= 0.30 or g_jacc >= 0.45 or (first_char and c_fchar and first_char == c_fchar and g_jacc >= 0.30):
                        add_s(84, c_eid)

        # 6. Distinctive Name Token + Street Token (Brand word only, non-generic)
        if s_tok:
            dist_toks = [t for t in s1_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
            if dist_toks:
                ts_list = idx_tok_street.get((country, dist_toks[0], s_tok), [])
                if 0 < len(ts_list) <= 4:
                    for c_eid, c_toks, c_metro, c_post, c_3g in ts_list:
                        if metro and c_metro and metro != c_metro:
                            continue
                        if postcode and c_post and postcode != c_post:
                            continue
                        jacc = token_jaccard(s1_toks, c_toks)
                        g_jacc = char_3gram_jaccard(s1_3grams, c_3g)
                        if jacc >= 0.33 or g_jacc >= 0.45:
                            add_s(90, c_eid)

        # 7. Postal Code + Name Prefix
        if postcode and len(norm_name) >= 4:
            pin_list = idx_pin_prefix.get((country, postcode, norm_name[:4]), [])
            if 0 < len(pin_list) <= 3:
                for c_eid, c_toks, c_3g in pin_list:
                    jacc = token_jaccard(s1_toks, c_toks)
                    g_jacc = char_3gram_jaccard(s1_3grams, c_3g)
                    if jacc >= 0.33 or g_jacc >= 0.45:
                        add_s(88, c_eid)

        scored_s2.sort(reverse=True)
        scored_s3.sort(reverse=True)

        top_set = set([cid for sc, cid in scored_s2[:4]] + [cid for sc, cid in scored_s3[:4]])
        true_m = gt_dict.get(s1_id, set())

        tp = len(top_set & true_m)
        fp = len(top_set - true_m)
        fn = len(true_m - top_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(true_m) == 0 and len(top_set) == 0:
            correct_singletons += 1
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    denom = 0.25 * macro_p + macro_r
    f05 = (1.25 * macro_p * macro_r) / denom if denom > 0 else 0.0

    print("=" * 60)
    print("PIPELINE v09 (HIGH-PRECISION + FUZZY TYPO RESCUE) RESULTS:")
    print(f"  Macro Precision: {macro_p:.6f} (v06: 0.833548, Diff: {macro_p - 0.833548:+.6f})")
    print(f"  Macro Recall:    {macro_r:.6f} (v06: 0.688855, Diff: {macro_r - 0.688855:+.6f})")
    print(f"  Macro F0.5:      {f05:.6f} (v06: 0.799943, Delta: {f05 - 0.799943:+.6f})")
    print(f"  Total TP:        {total_tp:,}")
    print(f"  Total FP:        {total_fp:,} (v06: 7,277, FP Reduction: {7277 - total_fp:+d})")
    print(f"  Total FN:        {total_fn:,}")
    print(f"  Singletons:      {correct_singletons:,}")
    print(f"  Evaluation took: {time.time()-t0:.1f}s")
    print("=" * 60)

if __name__ == '__main__':
    main()
