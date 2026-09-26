import csv, os, re, time
from collections import defaultdict
import numpy as np

# State and Region definitions
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

PUNCT_CHARS = ".,()[]{}:;\"'/?!@#$%^&*-+=~`|\\<>"
CHAR_MAP = str.maketrans(PUNCT_CHARS, " " * len(PUNCT_CHARS))

def fast_norm(text: str) -> str:
    if not text: return ""
    s = str(text).casefold()
    for dom in ['.com', '.org', '.net', '.in', '.co', '.fr', 'www.']:
        s = s.replace(dom, ' ')
    return " ".join(s.translate(CHAR_MAP).split())

def extract_state(norm_addr: str, country: str) -> str:
    if not norm_addr: return ""
    if country == 'india':
        for st in INDIAN_STATES:
            if st in norm_addr:
                return st
    elif country == 'us':
        tokens = set(norm_addr.split())
        for st in US_STATES:
            if st in tokens:
                return st
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
    precisions = []
    recalls = []
    f05_scores = []
    
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
            if (0.25 * p + r) > 0:
                f05 = (1.25 * p * r) / (0.25 * p + r)
            else:
                f05 = 0.0
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f05_scores)

def main():
    print("Loading Ground Truth and 5,000 Validation Entities...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    val_path = os.path.join(script_dir, 'val_indices.npy')
    gt_path = os.path.join(repo_root, 'dataset', 'train', 'train_ground_truth.tsv')
    s1_path = os.path.join(repo_root, 'dataset', 'train', 'train_source1.tsv')
    s2_path = os.path.join(repo_root, 'dataset', 'train', 'train_source2.tsv')
    
    val_indices = set(np.load(val_path)[:5000])
    
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
    s1_names = []
    s1_addrs = []
    s1_ctrys = []
    s1_bnums = []
    s1_posts = []
    s1_states = []
    s1_atokens = []
    s1_cores = []
    
    idx_exact = defaultdict(list)
    idx_core = defaultdict(list)
    
    with open(s1_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for idx, row in enumerate(r):
            if idx in val_indices:
                s1_id = row[0]
                norm_name = fast_norm(row[1])
                raw_addr = row[2] if len(row) > 2 else ""
                norm_addr = fast_norm(raw_addr)
                country = fast_norm(row[3]) if len(row) > 3 else ""
                b_num = extract_primary_number(raw_addr)
                pcode = extract_postal_code(raw_addr)
                state = extract_state(norm_addr, country)
                atok = get_addr_tokens(norm_addr)
                core = extract_core_name(norm_name)
                
                pos = len(s1_ids)
                s1_ids.append(s1_id)
                s1_names.append(norm_name)
                s1_addrs.append(norm_addr)
                s1_ctrys.append(country)
                s1_bnums.append(b_num)
                s1_posts.append(pcode)
                s1_states.append(state)
                s1_atokens.append(atok)
                s1_cores.append(core)
                
                if norm_name and country:
                    idx_exact[(country, norm_name)].append(pos)
                if core and country and core != norm_name:
                    idx_core[(country, core)].append(pos)

    print(f"Indexed {len(s1_ids)} S1 validation entities.")

    # Two test strategies:
    # 1. Baseline: Match any exact/core with score >= 75 (old logic)
    # 2. Strict Address Gated: Reject state mismatch, reject postcode mismatch, require address overlap if both have addr
    preds_baseline = defaultdict(list)
    preds_gated = defaultdict(list)
    
    print("Streaming 500,000 S2 records against validation indices...")
    t0 = time.time()
    with open(s2_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 500000: break
            s2_id = row[0]
            norm_name = fast_norm(row[1])
            raw_addr = row[2] if len(row) > 2 else ""
            norm_addr = fast_norm(raw_addr)
            country = fast_norm(row[3]) if len(row) > 3 else ""
            if not norm_name or not country: continue
            
            b_num = extract_primary_number(raw_addr)
            pcode = extract_postal_code(raw_addr)
            state = extract_state(norm_addr, country)
            atok = get_addr_tokens(norm_addr)
            core = extract_core_name(norm_name)
            
            # Check Exact
            candidates = idx_exact.get((country, norm_name), [])
            for s1_i in candidates:
                s1_id = s1_ids[s1_i]
                # Baseline
                if len(preds_baseline[s1_id]) < 2:
                    preds_baseline[s1_id].append(s2_id)
                
                # Gated logic:
                # 1. State conflict
                if state and s1_states[s1_i] and state != s1_states[s1_i]:
                    continue
                # 2. Postal code conflict
                if pcode and s1_posts[s1_i] and pcode != s1_posts[s1_i]:
                    continue
                # 3. Building number conflict
                if b_num > 0 and s1_bnums[s1_i] > 0 and abs(b_num - s1_bnums[s1_i]) > 2:
                    continue
                # 4. If both have address tokens, do they overlap?
                if atok and s1_atokens[s1_i] and len(atok & s1_atokens[s1_i]) == 0:
                    # If neither bnum nor postcode matches, zero address overlap is a false merge
                    if not (pcode and s1_posts[s1_i] and pcode == s1_posts[s1_i]) and \
                       not (b_num > 0 and s1_bnums[s1_i] > 0 and b_num == s1_bnums[s1_i]):
                        continue
                if len(preds_gated[s1_id]) < 2:
                    preds_gated[s1_id].append(s2_id)
                    
            # Check Core Name
            if core and core != norm_name:
                candidates_c = idx_core.get((country, core), [])
                for s1_i in candidates_c:
                    s1_id = s1_ids[s1_i]
                    # Baseline
                    if len(preds_baseline[s1_id]) < 2:
                        preds_baseline[s1_id].append(s2_id)
                        
                    # Gated logic:
                    # Single word core name requires strong address proof!
                    is_single_word = len(core.split()) == 1
                    
                    if state and s1_states[s1_i] and state != s1_states[s1_i]:
                        continue
                    if pcode and s1_posts[s1_i] and pcode != s1_posts[s1_i]:
                        continue
                    if b_num > 0 and s1_bnums[s1_i] > 0 and abs(b_num - s1_bnums[s1_i]) > 2:
                        continue
                    if is_single_word:
                        # Must have positive address proof
                        has_bnum_match = (b_num > 0 and s1_bnums[s1_i] > 0 and b_num == s1_bnums[s1_i])
                        has_post_match = (pcode and s1_posts[s1_i] and pcode == s1_posts[s1_i])
                        has_tok_match = len(atok & s1_atokens[s1_i]) > 0
                        if not (has_bnum_match or has_post_match or has_tok_match):
                            continue
                    else:
                        if atok and s1_atokens[s1_i] and len(atok & s1_atokens[s1_i]) == 0:
                            if not (pcode and s1_posts[s1_i] and pcode == s1_posts[s1_i]) and \
                               not (b_num > 0 and s1_bnums[s1_i] > 0 and b_num == s1_bnums[s1_i]):
                                continue
                    if len(preds_gated[s1_id]) < 2:
                        preds_gated[s1_id].append(s2_id)

    print(f"Streaming done in {time.time()-t0:.1f}s.")
    
    p_b, r_b, f_b = compute_macro_f05(preds_baseline, gt_dict, s1_ids)
    p_g, r_g, f_g = compute_macro_f05(preds_gated, gt_dict, s1_ids)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON ON GROUND TRUTH:")
    print("="*60)
    print(f"BASELINE (UN-GATED):")
    print(f"  Precision: {p_b*100:.2f}% | Recall: {r_b*100:.2f}% | Macro F0.5: {f_b:.4f}")
    print(f"STRICT ADDRESS GATED:")
    print(f"  Precision: {p_g*100:.2f}% | Recall: {r_g*100:.2f}% | Macro F0.5: {f_g:.4f}")
    print("="*60)
    print(f"IMPROVEMENT: Delta F0.5 = {f_g - f_b:+.4f} (Precision: {p_g*100 - p_b*100:+.2f}%)")
    print("="*60)

if __name__ == '__main__':
    main()
