import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, unicodedata, string, csv
from collections import defaultdict
import numpy as np
import pandas as pd

print("=" * 60)
print("EXPERIMENT v02: Normalization - Unicode NFKD (Diacritic Removal)")
print("=" * 60)

train_dir = 'dataset/train'
s1_path = os.path.join(train_dir, 'train_source1.tsv')
s2_path = os.path.join(train_dir, 'train_source2.tsv')
s3_path = os.path.join(train_dir, 'train_source3.tsv')
gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')

# Load GT
gt_dict = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        m = row['matched_entity_ids']
        gt_dict[row['source1_entity_id']] = set(m.strip().split(',')) if m and m.strip() else set()

# Fixed split
s1_records = []
with open(s1_path, 'r', encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        s1_records.append(row)

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
val_s1 = [s1_records[i] for i in val_indices]

# Precompute translation table
PUNCTUATION_CHARS = {
    i: 32
    for i in range(0x110000)
    if unicodedata.category(chr(i)).startswith('P') or chr(i) in string.punctuation
}

def normalize_text_nfkd(text: str) -> str | None:
    if text is None or pd.isna(text):
        return None
    # 1. NFKD decomposition to separate base letters from diacritics
    s = unicodedata.normalize('NFKD', str(text))
    # 2. Filter out non-spacing mark characters (Mn) like accents, umlauts
    s = "".join(c for c in s if unicodedata.category(c) != 'Mn')
    # 3. Unicode casefold
    s = s.casefold().translate(PUNCTUATION_CHARS)
    s = " ".join(s.split())
    return s if s else None

# Build index from S2 & S3
index = defaultdict(list)
for path in [s2_path, s3_path]:
    t0 = time.time()
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            bname = row.get("business_name")
            country = row.get("country")
            norm_name = normalize_text_nfkd(bname)
            norm_country = normalize_text_nfkd(country)
            if norm_name is not None and norm_country is not None and eid:
                index[(norm_country, norm_name)].append(eid)
    print(f"Indexed {path} in {time.time()-t0:.1f}s")

# Evaluate
precisions = []
recalls = []
correct_singletons = 0
included_entities = 0
total_tp = 0
total_fp = 0
total_fn = 0

for row in val_s1:
    eid = row['entity_id']
    norm_name = normalize_text_nfkd(row['business_name'])
    norm_country = normalize_text_nfkd(row['country'])

    if norm_name is not None and norm_country is not None:
        preds = set(index.get((norm_country, norm_name), []))
    else:
        preds = set()

    true_matches = gt_dict.get(eid, set())

    tp = len(preds.intersection(true_matches))
    fp = len(preds - true_matches)
    fn = len(true_matches - preds)
    total_tp += tp
    total_fp += fp
    total_fn += fn

    if len(true_matches) == 0 and len(preds) == 0:
        correct_singletons += 1
    else:
        included_entities += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

macro_prec = float(np.mean(precisions))
macro_rec = float(np.mean(recalls))
denom = 0.25 * macro_prec + macro_rec
f05 = (1.25 * macro_prec * macro_rec) / denom if denom > 0 else 0.0

print("\n" + "=" * 60)
print(f"EXPERIMENT v02 (Unicode NFKD) RESULTS:")
print("=" * 60)
print(f"Macro Precision: {macro_prec:.6f} (baseline: 0.390579, delta: {macro_prec - 0.390579:+.6f})")
print(f"Macro Recall:    {macro_rec:.6f} (baseline: 0.214006, delta: {macro_rec - 0.214006:+.6f})")
print(f"F0.5 Score:      {f05:.6f} (baseline: 0.335256, delta: {f05 - 0.335256:+.6f})")
print(f"Total TP: {total_tp:,} (delta: {total_tp - 333339:+d})")
print(f"Total FP: {total_fp:,} (delta: {total_fp - 3973876:+d})")
print(f"Total FN: {total_fn:,} (delta: {total_fn - 1194528:+d})")
print(f"Correct Singletons: {correct_singletons:,}, Included: {included_entities:,}")
