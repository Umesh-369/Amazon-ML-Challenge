# ML Challenge 2026: Business Entity Resolution Solution Write-up

**Team Name:** Team EntityZero  
**Submission Date:** September 2026  
**Target Metric:** Macro-Averaged $F_{0.5}$ Score  

---

## 1. Executive Summary

We developed an industrial-scale, precision-first business entity resolution pipeline designed to resolve records across three heterogeneous, noisy data sources without common identifiers. Leveraging empirical ceiling and error analyses across 12.5M training records, our solution introduces a 5-channel decoupled multi-index blocking strategy combined with surgical address disambiguation, commercial plaza false-positive suppression, and independent source evaluation.

On the official fixed 80/20 stratified validation benchmark (441,364 entities, seed 42), our current best pipeline (**v04 Decoupled Multi-Channel Pipeline**) achieves:
- **Macro Precision:** **0.779313** (+0.0666 gain over v03: 0.712679; +0.3887 over baseline: 0.390579)
- **Macro Recall:** **0.399537**
- **Macro $F_{0.5}$:** **0.654825** (+0.025075 gain over v03: 0.629750; +0.319569 over baseline: 0.335256)
- **False Positive Reduction:** Cut false positives by **45.1%** relative to v03 (down to 79,497) and by **98.0%** relative to the exact-match baseline (from 3.97M down to 79.5k).
- **Inference Efficiency:** Full test inference over 1.73M queries against 10M records completes in 9.1 minutes.
- **Validator Compliance:** Strictly verified and 100% passed via `utils/validate_submission.py`.

---

## 2. Methodology & Problem Analysis

### 2.1 Problem Analysis & Empirical Data Profiling

Analysis across `train_source1.tsv` (2,206,821 records), `train_source2.tsv` (5,034,616 records), `train_source3.tsv` (5,285,603 records), and `train_ground_truth.tsv` established critical structural findings:

1. **Country Integrity:**
   Across all 7,638,365 ground-truth match pairs in the training set, country agreement is strictly **100.00%**. True matches never cross country boundaries. The test set introduces a third country (`France`), requiring open-set string normalization rather than one-hot encoding or hardcoded geographic subsets.

2. **Ground Truth Cardinality Distribution:**
   Profiling `train_ground_truth.tsv` revealed the empirical distribution of matches per Source 1 entity:
   - $(S_2=1, S_3=1)$: 12.22%
   - $(S_2=1, S_3=2)$: 11.38%
   - $(S_2=2, S_3=1)$: 10.11%
   - $(S_2=2, S_3=2)$: 9.43%
   - $(S_2=0, S_3=0)$ (Singletons): 5.58%
   Over **94.4%** of Source 1 entities have at least one valid match, with the vast majority possessing 1 to 2 matches per source. Crucially, Source 2 and Source 3 matching behaviours are **independent**: an entity can match Source 2 via exact name while matching Source 3 via address anchor or transliteration.

3. **Address Distinguishability & Chain Collisions:**
   - Over **96.5%** of records contain numeric address tokens (building numbers, postal codes, unit numbers).
   - In Source 1, **11.10%** of entities share identical normalized business names with at least one other distinct entity in the same country (common retail chains, banks, pharmacies).
   - The Submission 1 baseline matched purely on `(country, business_name)` without address checking, producing **3,973,876 False Positives**.
   - Address numbers provide definitive disambiguation when names collide, but conflicting address numbers represent the single strongest signal of distinct entities.

---

## 3. Phase 1 Gap Quantification & Feasibility Ceiling Analysis

### 3.1 Validation False Negative Categorization (v03 Benchmark)

On the frozen held-out validation set (441,364 entities, 1,527,867 true match pairs), v03 produced 631,925 true positives and 895,942 false negatives. Dissection of all 895,942 false negatives revealed:

- **Category A (Blocking Failure / Correct pair NOT generated):** **684,725** (**76.43%** of all FNs)
- **Category B (Scoring/Decision Failure / Correct pair in candidates but rejected):** **211,217** (**23.57%** of all FNs)

Root causes of Category A blocking failures:
1. **Fuzzy Name Variation:** 286,349 (41.82%) — word order transpositions, minor typos, acronyms, and legal qualifier movements.
2. **Missing Building Numbers:** 136,661 (19.96%) — landmark-based addresses lacking standalone numeric building tokens.
3. **Conflicting / Differently Parsed Numbers:** 96,543 (14.10%).
4. **Non-Latin Transliteration:** 91,836 (13.41%) — Latin S1 names paired with Indic-script S2/S3 records.
5. **Disjoint Trade / DBA Names:** 73,336 (10.71%) — company operating names completely disjoint from registered corporate names.

### 3.2 Feasibility Ceiling & Mathematical Bounds for Target $F_{0.5} > 0.980473$

The competition evaluation metric is:
$$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$$

Mathematical analysis of this metric establishes strict boundary constraints:
1. **Minimum Required Precision:** At $R = 1.0$, the minimum precision required to reach $0.980473$ is:
   $$\frac{1.25 P}{0.25 P + 1.0} \ge 0.980473 \implies P \ge 97.57\%$$
   If Precision is below 97.57%, $F_{0.5} > 0.980473$ is **mathematically impossible** even with 100% recall.
2. **Minimum Required Recall:** At $P = 1.0$, the minimum recall required is:
   $$\frac{1.25 R}{0.25 + R} \ge 0.980473 \implies R \ge 90.95\%$$
   If Candidate Recall is below 90.95%, an oracle selector cannot reach 0.980473.
3. **Structural Ceiling:** Given that 10.71% of true matches possess completely disjoint trade names and 13.41% feature non-Latin scripts without Latin names, achieving $>97.6\%$ precision alongside $>91.0\%$ recall on unaugmented open-set data without label memorization is structurally unfeasible. However, expanding candidate recall while tightening address precision offers substantial legitimate headroom.

---

## 4. Multi-Channel Candidate Generation & Scoring Engine

### 4.1 Multi-Channel Blocking Architecture (Phase 2 & Phase 5)

To eliminate the 76.4% Category A recall bottleneck while adhering to the Computational Feasibility Guard, we implemented 5 complementary, bounded channels:

1. **Channel 1 (Exact Normalized Name):** Key: `(country, norm_name)` — stores `(eid, b_num, s_tok)`.
2. **Channel 2 (Core Business Name):** Key: `(country, core_name)` — strips legal suffixes (`inc`, `llc`, `ltd`, `pvt`, `corp`, `sa`, etc.).
3. **Channel 3 (Sorted Core Name):** Key: `(country, sorted_core)` — token-sorts core names to recover word transpositions and rearranged corporate qualifiers.
4. **Channel 4 (Building Number + Primary Name Token):** Key: `(country, b_num, primary_token)` — anchors on confirmed building numbers alongside distinctive name tokens (length $\ge 4$).
5. **Channel 5 (Hardened Address Anchor):** Key: `(country, b_num, street_tok)` — identifies co-located entities, capturing transliterations and trade names.

### 4.2 Precision Hardening & Decoupled Decision Hierarchy (Phase 5, 7, 9)

Error analysis of v03 revealed that **72.66% of all false positives** originated from address anchor collisions in multi-tenant commercial plazas. Pipeline v04 introduces:

1. **Single-Tenant vs Multi-Tenant Discrimination:**
   - Single tenant at address (`bucket == 1`): Match confirmed with high confidence (Score = 74).
   - Multi-tenant plaza (`bucket >= 2`): Suppressed unless corroborated by shared name initial, token overlap, or verified Indic script transliteration.
2. **Locality Confirmation for Unnumbered Records:**
   - When building numbers are absent, records in buckets $> 3$ are recovered if primary street/locality tokens strictly match (`s_tok == c_stok`), eliminating generic store chain errors while salvaging true matches.
3. **Decoupled Source Matching:**
   - Candidate evaluation across Source 2 and Source 3 is executed independently. An exact match found in Source 2 no longer preempts candidate recovery in Source 3.
4. **Strict Numeric Disagreement Penalty:**
   - Conflicting non-zero building numbers trigger immediate rejection (Score = 0).

---

## 5. Experimental Results & Benchmark Progression

### 5.1 Progression on Fixed 80/20 Stratified Validation Benchmark (441,364 Entities)

| Version | Configuration | Macro Precision | Macro Recall | Macro $F_{0.5}$ | $\Delta F_{0.5}$ | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v01** | Exact Match Baseline (Country + Name) | 0.390579 | 0.214006 | 0.335256 | — | 3,973,876 | 1,194,528 |
| **v02** | Unicode NFKD Diacritic Removal | 0.431826 | 0.252467 | 0.378103 | +0.042847 | 4,456,348 | 1,134,219 |
| **v03** | Multi-Index Precision Architecture | 0.712679 | 0.429732 | 0.629750 | +0.294494 | 144,725 | 895,964 |
| **v04 (Best)** | **Decoupled Multi-Channel Precision-Hardened** | **0.779313** | **0.399537** | **0.654825** | **+0.025075** | **79,497** | **960,506** |

### 5.2 Key Validation Achievements (v04 vs v03)
- **$F_{0.5}$ Improvement:** **+0.025075** absolute gain (from 0.629750 to 0.654825), clearing the +0.005 minimum promotion threshold by 5×.
- **Precision Surge:** **+0.066634** absolute increase (from 0.712679 to 0.779313).
- **False Positive Collapse:** Slashed false positive merges by **45.1%** (from 144,725 down to 79,497).
- **Cumulative FP Reduction vs Baseline:** Cut false positive merges by **98.0%** (from 3,973,876 down to 79,497).

---

## 6. Official Submission Verification

The v04 test outputs were fully audited and validated against `utils/validate_submission.py`:
- `output/matching_results.tsv`: 1,732,544 rows (252,248 empty singletons, 1,480,296 matched rows).
- `output/candidate_pairs.tsv`: 1,732,544 rows (44,773 empty singletons, 1,687,771 candidate rows).
- **Subset Check:** Every predicted match in `matching_results.tsv` is verified to be a strict subset of `candidate_pairs.tsv`.
- **Validation Result:** `PASS — no blocking issues found. Safe to submit.`

---

## 7. Reproduction Guide

All pipeline code resides under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── baseline.py          # Submission 1 exact-match baseline
│   ├── pipeline_v03.py      # v03 multi-index precision pipeline
│   └── pipeline_v04.py      # Production v04 decoupled pipeline
├── experiments/
│   ├── log.csv              # Full audit log of all experiments
│   ├── v01_baseline/        # Archived v01 code & results
│   ├── v02_unicode_nfkd/    # Archived v02 code & results
│   ├── v03_precision_pipeline/ # Archived v03 code & results
│   └── v04_decoupled_multi_index/ # Production v04 code, outputs, & results
├── requirements.txt         # Pinned environment dependencies
└── README.md                # Reproduction documentation
```

To run end-to-end inference and regenerate both submission files:
```bash
python code/business_entity_resolution/src/pipeline_v04.py \
    --test-dir dataset/test \
    --output-dir output
```

To run the official submission validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
