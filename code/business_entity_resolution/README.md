# Amazon ML Challenge 2026: Business Entity Resolution

## Production Solution — Decoupled Multi-Channel Precision-Hardened Pipeline (v04)

### 1. Overview
In large-scale commercial platforms, entity records arrive from noisy, independent sources lacking shared identifiers. This package implements a high-performance, precision-oriented Machine Learning solution to resolve business entities from **Source 1** against **Source 2** and **Source 3**.

The primary evaluation metric is **Macro-Averaged $F_{0.5}$**, which weights precision twice as heavily as recall:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### 2. Architecture & Key Innovations (v04)
1. **Universal Fast C-Level Normalization:**
   - Precomputed Unicode C translation table eliminating diacritics (`é` -> `e`, `ä` -> `a`, `ç` -> `c`), replacing all punctuation with whitespace, and collapsing whitespace.
   - Built to handle multilingual datasets including `US`, `India`, and open-set `France`.

2. **5-Channel Decoupled Candidate Generation (Blocking):**
   - **Channel 1 (Exact Name):** `(country, normalized_business_name)`
   - **Channel 2 (Core Name):** `(country, core_business_name)` with legal suffixes (`inc`, `llc`, `ltd`, `pvt`, `corp`, etc.) stripped.
   - **Channel 3 (Sorted Core Name):** `(country, sorted_core_name)` recovering word transpositions and rearranged corporate qualifiers.
   - **Channel 4 (Building Number + Primary Name Token):** `(country, building_number, primary_name_token)` capturing corporate name extensions and branch variations at verified physical premises.
   - **Channel 5 (Hardened Address Anchor):** `(country, building_number, street_token)` which matches records where building numbers and primary street tokens match, recovering non-ASCII Indian script transliterations and acronyms.

3. **Decoupled Source Matching Hierarchy:**
   - Evaluates Source 2 and Source 3 candidate channels independently so that an exact match found in one source does not artificially suppress recall of valid transliterated or address-anchored matches in the other source.

4. **Surgical Precision Hardening & False Positive Suppression:**
   - **Commercial Plaza Discrimination:** Distinguishes single-tenant physical premises (`bucket == 1`) from multi-tenant plazas (`bucket >= 2`), which require name initial/token agreement or Indic script confirmation, cutting 45.1% of false positive merges.
   - **Unnumbered Locality Recovery:** When building numbers are absent, records in larger buckets are safely matched if primary locality/street tokens strictly match.
   - **Contradiction Penalty:** Conflicting building numbers trigger immediate rejection.

### 3. Environment & Dependencies
- Python 3.8+ (Tested on Python 3.11)
- Dependencies:
  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  ```
  Contents:
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `rapidfuzz>=3.0.0`
  - `xgboost>=3.0.0`
  - `jellyfish>=1.0.0`

### 4. Reproduction & Inference

To run the complete end-to-end pipeline and regenerate both output files (`output/matching_results.tsv` and `output/candidate_pairs.tsv`):

```bash
python code/business_entity_resolution/src/pipeline_v04.py \
  --test-dir dataset/test \
  --output-dir output
```

To run the fixed 80/20 stratified validation benchmark on the training data:

```bash
python code/business_entity_resolution/src/pipeline_v04.py --validate
```

### 5. Submission Validation

Verify that generated output files comply with all formatting, header, singleton, and candidate consistency rules:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### 6. Validation Results (Fixed 80/20 Stratified Split, Seed 42, 441,364 entities)

| Version | Description | Macro Precision | Macro Recall | $F_{0.5}$ Score | $\Delta F_{0.5}$ vs Baseline | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v01** | Submission 1 Exact-Match Baseline | 0.390579 | 0.214006 | 0.335256 | — | 3,973,876 | 1,194,528 |
| **v02** | Unicode NFKD Diacritic Removal | 0.431826 | 0.252467 | 0.378103 | +0.042847 | 4,456,348 | 1,134,219 |
| **v03** | Multi-Index Precision Architecture | 0.712679 | 0.429732 | 0.629750 | +0.294494 | 144,725 | 895,964 |
| **v04 (Current Best)** | **Decoupled Multi-Channel Precision-Hardened** | **0.779313** | **0.399537** | **0.654825** | **+0.319569** | **79,497** | **960,506** |

- **Official Validator Status:** `PASS — no blocking issues found. Safe to submit.`
- **False Positive Reduction:** Slashed false positive merges by **45.1%** relative to v03 (down to 79,497) and by **98.0%** relative to the baseline (from 3.97M down to 79.5k).
- **Candidate Recall:** Reaches **60.26%** with an average of 36.69 candidates per Source 1 entity.
