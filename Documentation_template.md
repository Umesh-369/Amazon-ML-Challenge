# ML Challenge 2026: Business Entity Resolution Solution Write-up

**Team Name:** Team EntityZero  
**Submission Date:** September 2026  
**Target Metric:** Macro-Averaged $F_{0.5}$ Score  

---

## 1. Executive Summary

We developed an industrial-scale, precision-first business entity resolution pipeline designed to resolve records across three heterogeneous, noisy data sources without common identifiers. Leveraging empirical ceiling and error analyses across 12.5M training records, our solution introduces a 3-channel multi-index blocking strategy combined with numeric address disambiguation and adaptive chain suppression. On the official fixed 80/20 stratified validation benchmark (441,364 entities), our solution achieves **Macro Precision = 0.712679**, **Macro Recall = 0.429732**, and **$F_{0.5} = 0.629750$** — an absolute improvement of **+0.294494** over the baseline ($F_{0.5} = 0.335256$) while reducing false positive merges by **96.4%** (from 3.97M to 144k). The entire pipeline executes test inference across 1.73M queries against 10M records in under 6.5 minutes and strictly passes the official submission validator.

---

## 2. Methodology

### 2.1 Problem Analysis & Exploratory Data Profiling

Deep exploratory data analysis across `train_source1.tsv` (2,206,821 records), `train_source2.tsv` (5,034,616 records), `train_source3.tsv` (5,285,603 records), and `train_ground_truth.tsv` revealed crucial structural insights:

1. **Country Integrity:**
   Across all 7,638,365 ground-truth match pairs in the training set, country agreement is strictly **100.00%**. True matches never cross country boundaries. The test set introduces a third country (`France`), requiring open-set string handling rather than one-hot encoding or country-specific hardcoding.

2. **Name Variation & Transliteration:**
   - Exact business name agreement between Source 1 and matched Source 2/3 records is only **10.87%**.
   - In India, ~38% of Source 2 records and ~29% of Source 3 records feature names transliterated into non-Latin regional scripts (Devanagari, Tamil, Telugu, Kannada, Bengali), whereas Source 1 is 100% Latin ASCII. String-similarity methods alone fail entirely on these pairs.
   - In both US and India, entities frequently appear as abbreviations (`Corp` vs. `Corporation`, `Pvt` vs. `Private`, `LLC`, `Ltd`), acronyms (`ZB` for `Zander Blue Co`), and trade/DBA names.

3. **Address Distinguishability & Chain Collisions:**
   - Over **96.5%** of Source 1 and Source 2/3 records contain numeric address tokens (house/building numbers, unit numbers, postal codes).
   - In Source 1, **11.10%** of entities share identical normalized business names with at least one other distinct entity in the same country (common retail chains, banks, pharmacies).
   - The Submission 1 baseline matched purely on `(country, business_name)` without address checking, producing **3,973,876 False Positives** on the validation set.
   - When building numbers and locality tokens are evaluated, entity collision drops to **0.01%**, demonstrating that address information provides near-perfect disambiguation when names collide.

### 2.2 Solution Strategy

To optimize the precision-weighted $F_{0.5}$ metric ($\beta = 0.5$, penalizing false merges 2× more than false negatives), we adopted a **Precision-First Multi-Index Architecture**:
- **Phase A (Universal Preprocessing):** A C-speed normalization translation table that maps Unicode diacritics (`é` $\to$ `e`, `ä` $\to$ `a`, `ç` $\to$ `c`), strips punctuation, and standardizes whitespace.
- **Phase B (Multi-Channel Blocking):** Constructing three complementary inverted indices to maximize candidate recall up to 83.4% while maintaining compact posting lists.
- **Phase C (Numeric & Locality Disambiguation):** Enforcing strict building number consistency and penalizing conflicting addresses to eradicate chain-store false positive explosions.
- **Phase D (Source-Aware Top-K Capping):** Constraining matches to top-scored candidates per source, aligning with empirical ground truth match cardinality distributions.

---

## 3. Candidate Generation (Blocking)

We decoupled candidate generation from final match decisions to ensure bounded candidate-set sizes with high recall:

1. **Channel 1 (Exact Normalized Name):**
   - Key: `(country, normalized_business_name)`
   - Captures clean matches across all sectors.

2. **Channel 2 (Core Business Name):**
   - Key: `(country, core_business_name)`
   - Strips statutory legal suffixes (`inc`, `llc`, `ltd`, `pvt`, `private`, `corp`, `corporation`, `company`, `llp`, `sa`, `sarl`, `sas`) and token-sorts names to handle word transpositions and legal entity changes.

3. **Channel 3 (Address Anchor Key):**
   - Key: `(country, primary_building_number, primary_street_token)`
   - Identifies businesses situated at the exact same physical building and street, directly recovering Indian regional script transliterations and acronyms without cross-lingual translation.

**Blocking Statistics (Full Test Set, 1.73M queries vs 10M records):**
- Total unique keys indexed:
  - Exact Name Keys: 7,442,010
  - Core Name Keys: 3,329,054
  - Address Keys: 3,036,993
- Total candidate links generated: 39,906,851 (average 23.03 candidates per Source 1 entity).
- Reduction Ratio: **99.9998%** compared to exhaustive cross-product.

---

## 4. Matching Model & Decision Rules

Rather than relying on opaque or resource-heavy transformer embeddings (which violate memory bounds on 10M-record scale and fail on cross-lingual transliteration), we designed a deterministic, highly interpretable decision hierarchy:

### Decision Rules:
1. **Rule 1 (Exact Name Match):**
   - If S1 and candidate both contain building numbers:
     - If numbers match: **Match confirmed** (Priority = 100).
     - If numbers strictly conflict: **Reject** (Score = 0).
   - If numbers are absent in one or both records:
     - Match confirmed only if key bucket size $\le 3$ (uncommon/unique business name).
     - If bucket size $> 3$ (common chain name like Subway/Shell): **Suppressed** unless independent address tokens confirm.

2. **Rule 2 (Core Name Match):**
   - Evaluated when exact name match is absent.
   - Requires identical core name (length $\ge 4$ characters) AND building number agreement (or unique bucket).

3. **Rule 3 (Address Anchor Match):**
   - Evaluated when building number and street token strictly match.
   - Recovers cross-lingual transliterations and acronyms.
   - Caps bucket size $\le 6$ to eliminate commercial multi-tenant plazas without distinctive signals.

4. **Source-Aware Matching Constraints:**
   - Predictions are grouped by source prefix (`S2-` and `S3-`), sorted by priority score, and capped at **top-2 matches per source**.
   - Output lists are deduplicated and lexicographically sorted.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance Progression (80/20 Stratified Split)

The validation benchmark strictly follows the competition methodology (80/20 stratified split of `train_source1.tsv`, seed 42, 441,364 held-out entities):

| Experiment | Configuration | Macro Precision | Macro Recall | Macro $F_{0.5}$ | $\Delta F_{0.5}$ | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v01** | Exact Baseline (Country + Name) | 0.390579 | 0.214006 | 0.335256 | — | 3,973,876 | 1,194,528 |
| **v02** | Unicode NFKD Normalization | 0.431826 | 0.252467 | 0.378103 | +0.042847 | 4,456,348 | 1,134,219 |
| **v03 (Best)** | **Multi-Index Precision Pipeline** | **0.712679** | **0.429732** | **0.629750** | **+0.294494** | **144,725** | **895,964** |

- **True Positives:** Increased from 333,339 to **631,903** (+89.6% increase).
- **False Positives:** Dropped from 3,973,876 to **144,725** (**96.4% reduction**).
- **Correct Singletons:** Increased from 15,410 to **17,110**.

### 5.2 Error Analysis

- **Remaining False Positives (144k):**
  Primarily occur in commercial office complexes / shopping centres where distinct small enterprises share the exact building number and address line without sufficient distinctive name tokens.
- **Remaining False Negatives (895k):**
  Comprise records where both business name has severe spelling/transliteration distortion AND the address is missing building numbers (e.g. landmark-only descriptions like "Near Temple").
- **Feasibility Ceiling Analysis:**
  Upper-bound recall with broad multi-channel blocking reaches ~97%. Because F0.5 heavily penalizes false merges ($2\times$ over recall), threshold tuning balances precision ($>0.71$) against recall ($>0.43$). Achieving $F_{0.5} > 0.98$ on this noisy dataset without ground-truth label leakage is structurally unfeasible due to inherent address noise and cross-lingual ambiguity.

---

## 6. Conclusion

Our solution demonstrates that principled data profiling, multi-channel inverted indexing, and numeric address disambiguation achieve dramatic improvements over baseline approaches on ultra-large entity resolution benchmarks. By cutting false positive merges by 96.4% and doubling true positives, our pipeline boosts validation $F_{0.5}$ from 0.335256 to **0.629750** (+0.2945 gain), runs in under 6.5 minutes on 10 million records, and passes all official submission checks.

---

## Appendix

### A. Code Artefacts & Reproduction Guide

All pipeline code resides under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── baseline.py          # Original exact-match baseline
│   └── pipeline_v03.py      # Production precision-first pipeline
├── experiments/
│   ├── log.csv              # Full experiment audit log
│   ├── v01_baseline/        # Archived v01 code & outputs
│   ├── v02_unicode_nfkd/    # Archived v02 code & outputs
│   └── v03_precision_pipeline/ # Archived v03 code & outputs
├── requirements.txt         # Pinned environment dependencies
└── README.md                # Comprehensive reproduction guide
```

To reproduce the submission files from scratch:
```bash
python code/business_entity_resolution/src/pipeline_v03.py \
    --test-dir dataset/test \
    --output-dir output
```

To validate outputs against the official submission validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
