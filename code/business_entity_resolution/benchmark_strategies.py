import csv, os, re, time
from collections import defaultdict
import numpy as np

# Ground truth & validation data
val_path = 'code/business_entity_resolution/val_indices.npy'
gt_path = 'dataset/train/train_ground_truth.tsv'
s1_path = 'dataset/train/train_source1.tsv'
s2_path = 'dataset/train/train_source2.tsv'
s3_path = 'dataset/train/train_source3.tsv'

val_indices = set(np.load(val_path)[:5000])

# Precomputed char maps
PUNCT_CHARS = ".,()[]{}:;\"'/?!@#$%^&*-+=~`|\\<>"
CHAR_MAP_V11 = str.maketrans(PUNCT_CHARS, " " * len(PUNCT_CHARS))

CHAR_MAP_V12 = {ord(c): 32 for c in PUNCT_CHARS}
diacritic_pairs = 'àa áa âa ãa äa åa èe ée êe ëe ìi íi îi ïi òo óo ôo õo öo ùu úu ûu üu ýy ÿy çc ñn'
for pair in diacritic_pairs.split():
    CHAR_MAP_V12[ord(pair[0])] = ord(pair[1])
    CHAR_MAP_V12[ord(pair[0].upper())] = ord(pair[1])

INDIAN_STATES = {
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa',
    'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala',
    'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland',
    'odisha', 'orissa', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana',
    'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal', 'delhi', 'chandigarh'
}

US_STATES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in',
    'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv',
    'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn',
    'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
}

INDIAN_STATE_ABBRS = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'ch': 'chandigarh', 'dl': 'delhi', 'ga': 'goa',
    'gj': 'gujarat', 'hr': 'haryana', 'hp': 'himachal pradesh', 'jh': 'jharkhand',
    'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra',
    'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram', 'nl': 'nagaland',
    'od': 'odisha', 'or': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan',
    'sk': 'sikkim', 'tn': 'tamil nadu', 'ts': 'telangana', 'tg': 'telangana',
    'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand', 'ua': 'uttarakhand',
    'wb': 'west bengal'
}

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

def fast_norm_v11(text: str) -> str:
    if not text: return ""
    s = str(text).casefold()
    for dom in ['.com', '.org', '.net', '.in', '.co', '.fr', 'www.']:
        s = s.replace(dom, ' ')
    return " ".join(s.translate(CHAR_MAP_V11).split())

def fast_norm_v12(text: str) -> str:
    if not text: return ""
    s = str(text).casefold()
    for dom in ['.com', '.org', '.net', '.in', '.co', '.fr', 'www.']:
        s = s.replace(dom, ' ')
    return " ".join(s.translate(CHAR_MAP_V12).split())

def extract_state_v11(norm_addr: str, country: str) -> str:
    if not norm_addr: return ""
    if country == 'india':
        for st in INDIAN_STATES:
            if st in norm_addr: return st
    elif country == 'us':
        tokens = set(norm_addr.split())
        for st in US_STATES:
            if st in tokens: return st
    return ""

def extract_state_v12(norm_addr: str, country: str) -> str:
    if not norm_addr: return ""
    tokens = norm_addr.split()
    token_set = set(tokens)
    if country == 'india':
        if 'orissa' in token_set: return 'odisha'
        for st in INDIAN_STATES:
            if st.replace(" ", "") in norm_addr.replace(" ", "") or all(w in token_set for w in st.split()):
                if st == 'goa' and 'goa' not in token_set: continue
                return st
        for t in tokens:
            if t in INDIAN_STATE_ABBRS: return INDIAN_STATE_ABBRS[t]
    elif country == 'us':
        ambiguous = {'ct', 'in', 'or', 'me', 'la', 'oh'}
        for t in tokens:
            if t in US_STATES and t not in ambiguous: return t
        for t in tokens[-2:]:
            if t in US_STATES: return t
    return ""

def extract_primary_number(addr: str) -> int:
    if not addr: return 0
    m = re.search(r'\b\d+\b', addr)
    if m:
        val = int(m.group(0).lstrip('0') or '0')
        return val if 0 < val < 10000000 else 0
    return 0

def extract_postal_code(addr: str) -> str:
    m = re.search(r'\b[1-9]\d{5}\b|\b\d{5}\b', addr)
    return m.group(0) if m else ""

def get_addr_tokens(norm_addr: str) -> set:
    return set([t for t in norm_addr.split() if t not in STOP_WORDS and len(t) >= 3 and not t.isdigit()])

def extract_core_name(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS]
    return " ".join(tokens)

def compute_macro_f05(preds: dict, gt: dict, val_ids: list) -> tuple:
    precisions, recalls, f05_scores = [], [], []
    for s1_id in val_ids:
        gt_set = gt.get(s1_id, set())
        pred_set = set(preds.get(s1_id, []))
        tp = len(pred_set & gt_set)
        
        if not pred_set and not gt_set:
            p, r, f05 = 1.0, 1.0, 1.0
        elif not pred_set and gt_set:
            p, r, f05 = 0.0, 0.0, 0.0
        elif pred_set and not gt_set:
            p, r, f05 = 0.0, 0.0, 0.0
        else:
            p = tp / len(pred_set)
            r = tp / len(gt_set)
            f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)
    return np.mean(precisions), np.mean(recalls), np.mean(f05_scores)

print("Loading Ground Truth and 5,000 Validation Entities...")
gt_dict = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if len(row) >= 2 and row[1].strip():
            gt_dict[row[0]] = set(row[1].strip().split(','))
        else:
            gt_dict[row[0]] = set()

s1_ids = []
s1_norm = []
s1_bnum = []
s1_post = []
s1_state_v11 = []
s1_state_v12 = []
s1_atok = []
s1_core = []

idx_exact = defaultdict(list)
idx_core = defaultdict(list)

with open(s1_path, 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for idx, row in enumerate(r):
        if idx in val_indices:
            s1_id = row[0]
            country = fast_norm_v12(row[3]) if len(row) > 3 else ""
            raw_addr = row[2] if len(row) > 2 else ""
            norm_addr_v11 = fast_norm_v11(raw_addr)
            norm_addr_v12 = fast_norm_v12(raw_addr)
            
            norm_name = fast_norm_v12(row[1])
            b_num = extract_primary_number(raw_addr)
            postcode = extract_postal_code(raw_addr)
            st_v11 = extract_state_v11(norm_addr_v11, country)
            st_v12 = extract_state_v12(norm_addr_v12, country)
            atok = get_addr_tokens(norm_addr_v12)
            core = extract_core_name(norm_name)
            
            pos = len(s1_ids)
            s1_ids.append(s1_id)
            s1_norm.append(norm_name)
            s1_bnum.append(b_num)
            s1_post.append(postcode)
            s1_state_v11.append(st_v11)
            s1_state_v12.append(st_v12)
            s1_atok.append(atok)
            s1_core.append(core)
            
            if norm_name and country:
                idx_exact[(country, norm_name)].append(pos)
            if core and country and core != norm_name:
                idx_core[(country, core)].append(pos)

print(f"Indexed {len(s1_ids)} S1 validation entities.")

# Test:
# 1. v11 Strict: score >= 85, top 2, strict state, strict bnum, strict empty addr
# 2. v11 Enhanced (v13): score >= 85, top 2, v12 state & diacritics, strict bnum, strict empty addr
# 3. v12 Top-1: score >= 82, strict top 1 per source
# 4. v12 Raw: score >= 82, top 2-3 per source

preds_v11 = defaultdict(list)
preds_v13 = defaultdict(list)
preds_v12_top1 = defaultdict(list)
preds_v12_raw = defaultdict(list)

def stream_source(source_path, prefix):
    print(f"Streaming {prefix} (first 1,000,000 records)...")
    t0 = time.time()
    with open(source_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 1000000: break
            cid = row[0]
            country = fast_norm_v12(row[3]) if len(row) > 3 else ""
            norm_name = fast_norm_v12(row[1])
            if not norm_name or not country: continue
            
            raw_addr = row[2] if len(row) > 2 else ""
            norm_addr_v11 = fast_norm_v11(raw_addr)
            norm_addr_v12 = fast_norm_v12(raw_addr)
            b_num = extract_primary_number(raw_addr)
            postcode = extract_postal_code(raw_addr)
            st_v11 = extract_state_v11(norm_addr_v11, country)
            st_v12 = extract_state_v12(norm_addr_v12, country)
            atok = get_addr_tokens(norm_addr_v12)
            core = extract_core_name(norm_name)
            
            # Exact channel
            for s1_i in idx_exact.get((country, norm_name), []):
                s1_id = s1_ids[s1_i]
                
                # Check v11 strict gate
                v11_pass = True
                if st_v11 and s1_state_v11[s1_i] and st_v11 != s1_state_v11[s1_i]:
                    v11_pass = False
                elif postcode and s1_post[s1_i] and postcode != s1_post[s1_i]:
                    v11_pass = False
                elif b_num > 0 and s1_bnum[s1_i] > 0 and abs(b_num - s1_bnum[s1_i]) > 2:
                    v11_pass = False
                elif not atok or not s1_atok[s1_i]:
                    if not (len(s1_norm[s1_i].split()) >= 3 and not postcode and not s1_post[s1_i]):
                        v11_pass = False
                elif not (atok & s1_atok[s1_i]) and not (b_num > 0 and s1_bnum[s1_i] > 0 and b_num == s1_bnum[s1_i]) and not (postcode and s1_post[s1_i] and postcode == s1_post[s1_i]):
                    v11_pass = False
                    
                if v11_pass and len([m for m in preds_v11[s1_id] if m.startswith(prefix)]) < 2:
                    preds_v11[s1_id].append(cid)
                    
                # Check v13 enhanced gate (strict bnum + strict empty addr, but modernized state + AP/TG)
                v13_pass = True
                if st_v12 and s1_state_v12[s1_i] and st_v12 != s1_state_v12[s1_i]:
                    if not ((st_v12 in {'telangana', 'andhra pradesh'}) and (s1_state_v12[s1_i] in {'telangana', 'andhra pradesh'}) and (atok & s1_atok[s1_i])):
                        v13_pass = False
                if postcode and s1_post[s1_i] and postcode != s1_post[s1_i]:
                    v13_pass = False
                if b_num > 0 and s1_bnum[s1_i] > 0 and abs(b_num - s1_bnum[s1_i]) > 2:
                    v13_pass = False
                if not atok or not s1_atok[s1_i]:
                    if not (len(s1_norm[s1_i].split()) >= 3 and not postcode and not s1_post[s1_i]):
                        v13_pass = False
                elif not (atok & s1_atok[s1_i]) and not (b_num > 0 and s1_bnum[s1_i] > 0 and b_num == s1_bnum[s1_i]) and not (postcode and s1_post[s1_i] and postcode == s1_post[s1_i]):
                    v13_pass = False
                    
                if v13_pass:
                    if len([m for m in preds_v13[s1_id] if m.startswith(prefix)]) < 2:
                        preds_v13[s1_id].append(cid)
                        
                # Top-1 vs Raw
                if v11_pass and len([m for m in preds_v12_top1[s1_id] if m.startswith(prefix)]) < 1:
                    preds_v12_top1[s1_id].append(cid)
                    
    print(f"  Streaming {prefix} done in {time.time()-t0:.1f}s.")

stream_source(s2_path, 'S2-')
stream_source(s3_path, 'S3-')

print("\n" + "="*70)
print("EVALUATION ON 5,000 GROUND TRUTH ENTITIES:")
print("="*70)
for name, preds in [("v11 Strict (Portal 0.670)", preds_v11),
                    ("v13 Enhanced Strict (v11 + Diacritics + State Abbrs)", preds_v13),
                    ("v12 Top-1 (Strict 1/source)", preds_v12_top1)]:
    p, r, f05 = compute_macro_f05(preds, gt_dict, s1_ids)
    n_preds = sum(len(v) for v in preds.values())
    print(f"{name}:")
    print(f"  Total Matches: {n_preds} (avg {n_preds/len(s1_ids):.2f}/entity)")
    print(f"  Precision:     {p*100:.2f}%")
    print(f"  Recall:        {r*100:.2f}%")
    print(f"  Macro F0.5:    {f05:.4f}")
    print("-" * 70)

