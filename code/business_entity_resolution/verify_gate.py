import csv, re

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

PUNCT_CHARS = ".,()[]{}:;\"'/?!@#$%^&*-+=~`|\\<>"
CHAR_MAP = str.maketrans(PUNCT_CHARS, ' ' * len(PUNCT_CHARS))

def fast_norm(text: str) -> str:
    if not text: return ''
    s = str(text).casefold()
    for dom in ['.com', '.org', '.net', '.in', '.co', '.fr', 'www.']:
        s = s.replace(dom, ' ')
    return ' '.join(s.translate(CHAR_MAP).split())

def extract_state(norm_addr: str, country: str) -> str:
    if not norm_addr: return ''
    if country == 'india':
        for st in INDIAN_STATES:
            if st in norm_addr: return st
    elif country == 'us':
        for t in norm_addr.split():
            if t in US_STATES: return t
    return ''

def get_addr_tokens(norm_addr: str) -> set:
    stop = {'the', 'and', 'of', 'in', 'at', 'on', 'near', 'opp', 'opposite', 'behind', 'floor',
            'road', 'street', 'st', 'rd', 'ave', 'avenue', 'lane', 'drive', 'dr', 'way', 'hwy', 'highway',
            'plot', 'block', 'sector', 'nagar', 'colony', 'apartment', 'apartments', 'bldg', 'building',
            'rue', 'avenue', 'boulevard', 'bd', 'chemin', 'place', 'flat', 'room', 'office', 'unit',
            'shop', 'cross', 'main', 'post', 'district', 'dist', 'taluk', 'tehsil', 'null'}
    return set([t for t in norm_addr.split() if t not in stop and len(t) >= 3 and not t.isdigit()])

def extract_bnum(addr: str) -> int:
    m = re.search(r'\b\d+\b', addr)
    return int(m.group(0).lstrip('0') or '0') if m else 0

def extract_post(addr: str) -> str:
    m = re.search(r'\b[1-9]\d{5}\b|\b\d{5}\b', addr)
    return m.group(0) if m else ''

# Test S1-280204013
s1_row = ['S1-280204013', 'Om Constructions Pvt Ltd', 'Karauli, Rajasthan, Karauli, Pani Ki Tanki Ke Pas Choubepada', 'India']
s1_norm = fast_norm(s1_row[1])
s1_addr = fast_norm(s1_row[2])
s1_ctry = fast_norm(s1_row[3])
s1_st = extract_state(s1_addr, s1_ctry)
s1_tok = get_addr_tokens(s1_addr)
s1_b = extract_bnum(s1_row[2])
s1_p = extract_post(s1_row[2])

print(f"S1: {s1_row[1]} | State: '{s1_st}' | Addr Tokens: {sorted(list(s1_tok))}")

candidates = [
    ('S2-981457115', 'Om Construction LLP', 'H.NO 595, SECTOR-15 PART-I GURGFON, Haryana', 'India'),
    ('S2-942852308', 'Om Consulting Private', 'MP / XIV / 777., PUTHIYAMADOM, MEENANGADI POST, Kerala', 'India'),
    ('S3-TRUE-HYPOTHETICAL', 'Om Construction Company', 'Pani Ki Tanki Ke Pas Choubepada, Karauli, Rajasthan', 'India')
]

for cid, cname, caddr, cctry in candidates:
    cnorm = fast_norm(cname)
    c_addr = fast_norm(caddr)
    c_st = extract_state(c_addr, fast_norm(cctry))
    c_tok = get_addr_tokens(c_addr)
    c_b = extract_bnum(caddr)
    c_p = extract_post(caddr)
    
    passed = True
    reason = 'OK'
    if c_st and s1_st and c_st != s1_st:
        passed = False
        reason = f"State mismatch ('{s1_st}' vs '{c_st}')"
    elif c_p and s1_p and c_p != s1_p:
        passed = False
        reason = f"Postcode mismatch ('{s1_p}' vs '{c_p}')"
    elif c_b > 0 and s1_b > 0 and abs(c_b - s1_b) > 2:
        passed = False
        reason = f"Bnum mismatch ({s1_b} vs {c_b})"
    elif not (c_tok & s1_tok) and not (c_b > 0 and s1_b > 0 and c_b == s1_b) and not (c_p and s1_p and c_p == s1_p):
        passed = False
        reason = 'Zero address token overlap'
        
    print(f"  Candidate {cid} ({cname}): Passed={passed} | Reason={reason}")
