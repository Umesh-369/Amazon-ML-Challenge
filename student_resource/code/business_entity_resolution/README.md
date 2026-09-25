# Submission 1 — Strict Exact-Match Baseline

## Amazon ML Challenge 2026: Business Entity Resolution

### 1. Overview
This package contains the implementation for **Submission 1 (Strict Exact-Match Baseline)**. It implements a deterministic, precision-oriented baseline that resolves entity matches between Source 1 and Sources 2 & 3 based purely on exact string equality of normalized business names and normalized countries.

### 2. Matching Logic & Rules
- **Business Name Normalization**:
  1. Unicode casefolding (`str.casefold()`).
  2. Stripping leading/trailing whitespace.
  3. Replacing all Unicode and ASCII punctuation characters with whitespace.
  4. Collapsing consecutive whitespace to a single space.
  5. Missing/null values are non-matchable.
- **Country Normalization**:
  1. Unicode casefolding.
  2. Stripping leading/trailing whitespace.
  3. Replacing punctuation characters with whitespace.
  4. Collapsing consecutive whitespace to a single space.
  5. Treated as an open-set string (supporting US, India, France, etc.).
  6. Missing/null values are non-matchable.
- **Matching Rule**:
  A candidate match is established if and only if:
  $$\text{normalized\_country}(S1) == \text{normalized\_country}(S2/S3)$$
  AND
  $$\text{normalized\_business\_name}(S1) == \text{normalized\_business\_name}(S2/S3)$$
- **Candidate Set & Matching Set**:
  In Submission 1, `candidate_pairs.tsv` and `matching_results.tsv` contain identical pairs, with IDs deduplicated and lexicographically sorted per Source 1 record.

### 3. Environment & Dependencies
- Python 3.8+
- Required packages:
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`

Install dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 4. Reproduction Instructions

To generate the submission outputs (`output/matching_results.tsv` and `output/candidate_pairs.tsv`), run:

```bash
python3 code/business_entity_resolution/src/baseline.py \
  --test-dir dataset/test \
  --output-dir output
```

*(Optional)* To also run the 80/20 stratified local validation before test inference:
```bash
python3 code/business_entity_resolution/src/baseline.py \
  --test-dir dataset/test \
  --output-dir output \
  --train-dir dataset/train \
  --validate
```

### 5. Submission Validation

Validate the output formatting against the official validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### 6. Baseline Validation Results (80/20 Stratified Split, Seed 42)
- **Held-out Split Size**: 441,364 Source 1 entities (20% of 2,206,821 total S1 entities)
- **Macro Precision**: 0.390579
- **Macro Recall**: 0.214006
- **F0.5 Score**: 0.335256
- **Correctly-identified Singletons**: 15,410
- **Entities Included in Macro Average**: 425,954
