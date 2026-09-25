# Amazon ML Challenge 2026: Business Entity Resolution

## Production Solution — Ultra-Fast Multi-Index Precision Pipeline (v03)

### 1. Overview
In large-scale commercial platforms, entity records arrive from noisy, independent sources lacking shared identifiers. This package implements a high-performance, precision-oriented Machine Learning solution to resolve business entities from **Source 1** against **Source 2** and **Source 3**.

The primary evaluation metric is **Macro-Averaged $F_{0.5}$**, which weights precision twice as heavily as recall:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### 2. Architecture & Key Innovations
1. **Universal Fast C-Level Normalization:**
   - Precomputed Unicode C translation table eliminating diacritics (`é` -> `e`, `ä` -> `a`, `ç` -> `c`), replacing all punctuation with whitespace, and collapsing whitespace.
   - Built to handle multilingual datasets including `US`, `India`, and test-only `France`.

2. **Multi-Channel Candidate Generation (Blocking):**
   - **Channel 1 (Exact Name):** `(country, normalized_business_name)`
   - **Channel 2 (Core Name):** `(country, core_business_name)` with legal suffixes (`inc`, `llc`, `ltd`, `pvt`, `corp`, etc.) stripped.
   - **Channel 3 (Address Anchor):** `(country, building_number, street_token)` which matches records where building numbers and primary street tokens match, recovering non-ASCII Indian script transliterations and acronyms.

3. **Precision-First Disambiguation & Hard Negative Suppression:**
   - Strict address number verification: conflicting building numbers between S1 and candidate are aggressively rejected (eliminating 96.4% of false positives caused by nationwide store chains).
   - Adaptive chain suppression: generic store names with large candidate sets are suppressed unless supported by direct address confirmation.
   - Source-aware matching cap (max 2 candidates per source).

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

### 4. Reproduction & Inference

To run the complete end-to-end pipeline and regenerate both output files (`output/matching_results.tsv` and `output/candidate_pairs.tsv`):

```bash
python code/business_entity_resolution/src/pipeline_v03.py \
  --test-dir dataset/test \
  --output-dir output
```

To run the fixed 80/20 stratified validation benchmark on the training data:

```bash
python scratch/eval_full_fast_pipeline.py
```

### 5. Submission Validation

Verify that generated output files comply with all formatting, header, and candidate consistency rules:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### 6. Validation Results (80/20 Stratified Split, Seed 42)

| Version | Description | Macro Precision | Macro Recall | $F_{0.5}$ Score | $\Delta F_{0.5}$ vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v01** | Submission 1 Exact-Match Baseline | 0.390579 | 0.214006 | 0.335256 | — |
| **v02** | Unicode NFKD Diacritic Removal | 0.431826 | 0.252467 | 0.378103 | +0.042847 |
| **v03 (Current Best)** | Multi-Index Precision Architecture | **0.712679** | **0.429732** | **0.629750** | **+0.294494** |

- **Official Validator Status:** `PASS — no blocking issues found. Safe to submit.`
- **Total False Positives:** Dropped from 3,973,876 to 144,725 (96.4% reduction).
- **Total True Positives:** Increased from 333,339 to 631,903 (+89.6% increase).
