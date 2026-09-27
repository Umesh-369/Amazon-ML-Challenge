import csv, os, sys, re, time
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

PUNCT_CHARS = ".,()[]{}:;\"'/?!@#$%^&*-+=~`|\\<>"
CHAR_MAP = {ord(c): 32 for c in PUNCT_CHARS}
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
    if not text: return ""
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

INDIAN_METROS = {
    'delhi': 'delhi', 'new delhi': 'delhi',
    'mumbai': 'mumbai', 'bombay': 'mumbai',
    'bangalore': 'bangalore', 'bengaluru': 'bangalore',
    'chennai': 'chennai', 'madras': 'chennai',
    'kolkata': 'kolkata', 'calcutta': 'kolkata',
    'hyderabad': 'hyderabad', 'secunderabad': 'hyderabad',
    'pune': 'pune', 'ahmedabad': 'ahmedabad'
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

def extract_state(norm_addr: str, country: str) -> str:
    if not norm_addr: return ""
    tokens = norm_addr.split()
    token_set = set(tokens)
    if country == 'india':
        if 'orissa' in token_set: return 'odisha'
        for st in INDIAN_STATES:
            # check whole word in address
            if st.replace(" ", "") in norm_addr.replace(" ", "") or all(w in token_set for w in st.split()):
                # ensure 'goa' isn't matching inside another word
                if st == 'goa' and 'goa' not in token_set:
                    continue
                return st
        for t in tokens:
            if t in INDIAN_STATE_ABBRS: return INDIAN_STATE_ABBRS[t]
    elif country == 'us':
        # Don't let street abbreviations like 'ct' (court) or words like 'in', 'or' trigger state
        ambiguous = {'ct', 'in', 'or', 'me', 'la', 'oh'}
        for t in tokens:
            if t in US_STATES and t not in ambiguous:
                return t
        # If ambiguous, check if it's the last or second to last token
        for t in tokens[-2:]:
            if t in US_STATES:
                return t
    return ""

def extract_metro(norm_addr: str) -> str:
    for word in norm_addr.split():
        if word in INDIAN_METROS: return INDIAN_METROS[word]
    return ""

def extract_core_name(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in GENERIC_INDUSTRY_WORDS]
    return " ".join(tokens)

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

def extract_first_street_token(norm_addr: str) -> str:
    for t in norm_addr.split():
        if len(t) >= 4 and t not in STOP_WORDS and not t.isdigit():
            return t
    return ""

def fast_soundex(word: str) -> str:
    if not word: return ""
    w = word.lower()
    first = w[0]
    table = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3', 'l': '4', 'm': '5', 'n': '5', 'r': '6'
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
    if len(s) < 3: return {s}
    return {s[i:i+3] for i in range(len(s) - 2)}

def token_jaccard(toks1: set, toks2: set) -> float:
    if not toks1 or not toks2: return 0.0
    u = len(toks1 | toks2)
    return len(toks1 & toks2) / u if u > 0 else 0.0

def char_3gram_jaccard(g1: set, g2: set) -> float:
    if not g1 or not g2: return 0.0
    u = len(g1 | g2)
    return len(g1 & g2) / u if u > 0 else 0.0

# 1. Load Ground Truth pairs
gt_pairs = []
with open(os.path.join(project_root, 'dataset/train/train_ground_truth.tsv'), 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if len(row) >= 2 and row[1]:
            s1_id = row[0]
            for m in row[1].split(','):
                gt_pairs.append((s1_id, m))
                if len(gt_pairs) >= 5000: break
        if len(gt_pairs) >= 5000: break

s1_needed = {p[0] for p in gt_pairs}
s2_needed = {p[1] for p in gt_pairs if p[1].startswith('S2-')}
s3_needed = {p[1] for p in gt_pairs if p[1].startswith('S3-')}

print(f"Loaded {len(gt_pairs)} true pairs across {len(s1_needed)} S1 entities.")

s1_data = {}
with open(os.path.join(project_root, 'dataset/train/train_source1.tsv'), 'r', encoding='utf-8') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] in s1_needed: s1_data[row[0]] = row

s2_data = {}
with open(os.path.join(project_root, 'dataset/train/train_source2.tsv'), 'r', encoding='utf-8') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] in s2_needed: s2_data[row[0]] = row

s3_data = {}
with open(os.path.join(project_root, 'dataset/train/train_source3.tsv'), 'r', encoding='utf-8') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] in s3_needed: s3_data[row[0]] = row

# Now let's diagnose each pair:
# Which channel hit? Did address gate pass? What score was computed?
hits_by_channel = Counter()
missed_by_all_channels = []
rejected_by_address_gate = []
low_score_rejected = []
accepted_matches = []

for s1_id, mid in gt_pairs:
    s1_row = s1_data.get(s1_id)
    cand_row = s2_data.get(mid) or s3_data.get(mid)
    if not s1_row or not cand_row: continue
    
    country = fast_norm(s1_row[3]) if len(s1_row) > 3 else ""
    c_country = fast_norm(cand_row[3]) if len(cand_row) > 3 else ""
    if country != c_country:
        missed_by_all_channels.append((s1_row, cand_row, "Country mismatch"))
        continue
        
    s1_norm = fast_norm(s1_row[1])
    c_norm = fast_norm(cand_row[1])
    s1_addr = fast_norm(s1_row[2])
    c_addr = fast_norm(cand_row[2])
    
    s1_b = extract_primary_number(s1_row[2])
    c_b = extract_primary_number(cand_row[2])
    s1_p = extract_postal_code(s1_row[2])
    c_p = extract_postal_code(cand_row[2])
    s1_st = extract_state(s1_addr, country)
    c_st = extract_state(c_addr, country)
    s1_atok = get_addr_tokens(s1_addr)
    c_atok = get_addr_tokens(c_addr)
    s1_m = extract_metro(s1_addr) if country == 'india' else ""
    c_m = extract_metro(c_addr) if country == 'india' else ""
    s1_stok = extract_first_street_token(s1_addr)
    c_stok = extract_first_street_token(c_addr)
    
    s1_core = extract_core_name(s1_norm)
    c_core = extract_core_name(c_norm)
    s1_toks = set([t for t in s1_norm.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
    c_toks = set([t for t in c_norm.split() if t not in GENERIC_INDUSTRY_WORDS and t not in STOP_WORDS])
    s1_sorted_core = " ".join(sorted(list(s1_toks))) if len(s1_toks) >= 2 else ""
    c_sorted_core = " ".join(sorted(list(c_toks))) if len(c_toks) >= 2 else ""
    s1_3g = get_char_3grams(s1_norm)
    c_3g = get_char_3grams(c_norm)
    
    # Check channels
    matched_channels = []
    if s1_norm == c_norm:
        matched_channels.append("Exact")
    if s1_norm.replace(" ", "") == c_norm.replace(" ", "") and len(s1_norm) >= 6:
        matched_channels.append("Spaceless")
    if s1_core and s1_core == c_core:
        matched_channels.append("Core")
    if s1_sorted_core and s1_sorted_core == c_sorted_core:
        matched_channels.append("SortedCore")
    if s1_b > 0 and s1_b == c_b and len(s1_norm) >= 4 and len(c_norm) >= 4 and s1_norm[:4] == c_norm[:4]:
        matched_channels.append("BnumPrefix")
    if s1_b > 0 and s1_b == c_b and s1_stok and (s1_stok == c_stok or (len(s1_stok) >= 4 and len(c_stok) >= 4 and fast_soundex(s1_stok) == fast_soundex(c_stok))):
        matched_channels.append("AddrAnchor")
    
    s1_dist = [t for t in s1_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
    c_dist = [t for t in c_toks if len(t) >= 5 and t not in GENERIC_INDUSTRY_WORDS]
    if s1_stok and s1_stok == c_stok and s1_dist and c_dist and s1_dist[0] == c_dist[0]:
        matched_channels.append("TokStreet")
    if s1_stok and s1_stok == c_stok and s1_dist and c_dist and fast_soundex(s1_dist[0]) == fast_soundex(c_dist[0]):
        matched_channels.append("SoundexStreet")
    if s1_p and s1_p == c_p and len(s1_norm) >= 4 and len(c_norm) >= 4 and s1_norm[:4] == c_norm[:4]:
        matched_channels.append("PINPrefix")
    if s1_dist and c_dist and s1_dist[0] == c_dist[0] and s1_st and s1_st == c_st:
        matched_channels.append("BrandState")
    s1_nums = [int(n) for n in re.findall(r'\b\d{6,10}\b', s1_row[2]) if len(n) in (6, 7, 10)]
    c_nums = [int(n) for n in re.findall(r'\b\d{6,10}\b', cand_row[2]) if len(n) in (6, 7, 10)]
    if s1_nums and c_nums and (set(s1_nums) & set(c_nums)):
        matched_channels.append("RegNumber")
        
    for ch in matched_channels:
        hits_by_channel[ch] += 1
        
    if not matched_channels:
        missed_by_all_channels.append((s1_row, cand_row))
        continue
        
    # Check address gating in v11+
    # 1. State conflict veto
    if c_st and s1_st and c_st != s1_st:
        is_ap_tg = (c_st in {'telangana', 'andhra pradesh'}) and (s1_st in {'telangana', 'andhra pradesh'})
        if is_ap_tg and (c_atok & s1_atok):
            pass
        else:
            rejected_by_address_gate.append((s1_row, cand_row, f"State mismatch: '{s1_st}' vs '{c_st}'"))
            continue
    # 2. Metro conflict veto
    if country == 'india' and c_m and s1_m and c_m != s1_m:
        rejected_by_address_gate.append((s1_row, cand_row, f"Metro mismatch: '{s1_m}' vs '{c_m}'"))
        continue
    # 3. Postal code conflict veto
    if c_p and s1_p and c_p != s1_p:
        rejected_by_address_gate.append((s1_row, cand_row, f"Postcode mismatch: '{s1_p}' vs '{c_p}'"))
        continue
    # 4. Building number conflict veto (SAFE: only if not sharing apartment/complex/locality tokens)
    if c_b > 0 and s1_b > 0 and abs(c_b - s1_b) > 2:
        shared_toks = c_atok & s1_atok
        if len(shared_toks) < 2 and token_jaccard(c_atok, s1_atok) < 0.25:
            rejected_by_address_gate.append((s1_row, cand_row, f"Bnum mismatch: {s1_b} vs {c_b}"))
            continue
    # 5. Address overlap proof
    has_addr_proof = False
    if c_b > 0 and s1_b > 0 and c_b == s1_b:
        has_addr_proof = True
    elif c_p and s1_p and c_p == s1_p:
        has_addr_proof = True
    elif c_atok and s1_atok and (c_atok & s1_atok):
        has_addr_proof = True
    elif not c_atok or not s1_atok:
        if s1_norm == c_norm or (s1_core and s1_core == c_core) or (s1_sorted_core and s1_sorted_core == c_sorted_core) or (s1_norm.replace(' ', '') == c_norm.replace(' ', '')):
            has_addr_proof = True
            
    if not has_addr_proof:
        rejected_by_address_gate.append((s1_row, cand_row, "No address overlap proof"))
        continue
        
    score = 90
    accepted_matches.append((s1_row, cand_row, score))

print("="*70)
print(f"DIAGNOSTIC REPORT ON {len(gt_pairs)} TRUE GROUND TRUTH PAIRS:")
print("="*70)
print(f"Accepted Matches:             {len(accepted_matches)} ({len(accepted_matches)/len(gt_pairs)*100:.2f}%)")
print(f"Missed by ALL Inverted Keys:  {len(missed_by_all_channels)} ({len(missed_by_all_channels)/len(gt_pairs)*100:.2f}%)")
print(f"Rejected by Address Gating:   {len(rejected_by_address_gate)} ({len(rejected_by_address_gate)/len(gt_pairs)*100:.2f}%)")
print("="*70)

print("\n--- ADDRESS REJECTION REASON BREAKDOWN ---")
rejection_reasons = Counter([r[2].split(':')[0] for r in rejected_by_address_gate])
for r, cnt in rejection_reasons.most_common():
    print(f"  {r:<30}: {cnt} ({cnt/len(rejected_by_address_gate)*100:.1f}%)")

print("\n--- SAMPLE MISSED BY ALL INVERTED KEYS (10 Samples) ---")
for s1, cand in missed_by_all_channels[:10]:
    print(f"  S1: '{s1[1]}' | ADDR: '{s1[2]}'")
    print(f"  M : '{cand[1]}' | ADDR: '{cand[2]}'")
    print("  " + "-"*50)
print("="*70)
