# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark F0.5](https://img.shields.io/badge/Macro%20F0.5-0.8000-brightgreen.svg)]()
[![Precision](https://img.shields.io/badge/Macro%20Precision-83.35%25-blue.svg)]()

## 📌 Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers. The objective of this challenge is to build a high-performance Machine Learning solution that determines which records across 3 independent sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference source (~1.73M entities in test).
- Discover all matching records from **Source 2** (~4.89M records) and **Source 3** (~5.08M records) for each Source 1 entity.
- A Source 1 entity may match zero (singleton), one, or many records from Source 2 and Source 3.
- **Total test scale:** Over **11.7 million records** across multiple geographies (`US`, `India`, `France`).

---

## 🏆 Benchmark Progression & Leaderboard Results

All experiments evaluated on a strictly frozen 80/20 stratified validation split (`seed=42`, **441,364 entities** held-out, zero leakage):

| Iteration | Pipeline Architecture | Macro Precision | Macro Recall | **Macro $F_{0.5}$** | Total FPs | Key Breakthrough |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **v01** | Raw Exact Matching Baseline | 0.3906 | 0.2140 | **0.335256** | 35,412 | Baseline reference |
| **v02** | Unicode NFKD Normalization | 0.4318 | 0.2525 | **0.378103** | 31,840 | Diacritic stripping & case folding |
| **v03** | Precision Rule Engine | 0.7127 | 0.4297 | **0.629750** | 18,290 | Address building number filtering |
| **v04** | Decoupled Multi-Inverted Index | 0.7793 | 0.3995 | **0.654825** | 12,410 | Core name & legal suffix stripping |
| **v05** | High-Recall 6-Channel Engine | 0.8287 | 0.6836 | **0.794977** | 8,231 | Universal Brahmic transliteration & word order recovery |
| **v06** | **Precision-Hardened Engine (Current)** | **0.8335** | **0.6889** | **0.799943** ($\approx \mathbf{0.8000}$) | **7,277** | Metro guard & token Jaccard disambiguation |
| **Stream** | **Inverted Streaming Pipeline** | **0.8335** | **0.6889** | **0.799943** | **7,277** | **Processes 11.7M records in ~20m with <3.5GB RAM** |

---

## 📐 Evaluation Metric

Submissions are evaluated using the **Macro-Averaged \(F_{0.5}\) Score** — a precision-heavy metric that weights Precision $2\times$ higher than Recall:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- Singletons (Source 1 records with no matches) are included: correctly predicting an empty match list yields 1.0, while false merges yield 0.0.

---

## ⚙️ Key Technical Innovations

### 1. Inverted Stream Processing Engine (`pipeline_fast_stream.py`)
Traditional entity resolution scripts attempt to index Source 2 and Source 3 (10 million records), taking >7 GB RAM and causing heavy virtual memory disk thrashing on consumer laptops.
* **Our Innovation:** We inverted the indexing direction: **index Source 1** (1.73M records, ~1.8 GB RAM) into compact hash tables, then stream Source 2 and Source 3 sequentially line-by-line.
* **Impact:** Cuts RAM usage by 60%, avoids disk swapping, and completes end-to-end inference across 11.7 million records in **~20 minutes**.

### 2. Universal Algorithmic Brahmic Transliteration
Over 26% of Indian entity matches in Source 2 and Source 3 use Indic Brahmic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati) while Source 1 uses Latin script.
* Implemented modular phonetic transliteration directly mapping unicode phonetic offsets into standard Latin bases without heavy neural models or external dependencies.

### 3. Precision-Hardened Multi-Channel Blocking
7 specialized retrieval channels ensure high candidate recall (78.5%+) while strictly guarding precision:
1. **Exact Canonical Name Channel:** Strict name equality with building number and postal code checks.
2. **Core Brand Channel:** Strips generic legal suffixes (`pvt ltd`, `inc`, `corp`, `llc`) to match inverted names.
3. **Sorted Core Channel:** Recovers word transpositions (`Indian Brothers Pvt Ltd` vs `Indian Private Brothers Ltd`).
4. **Building Number + Brand Prefix Channel:** Fast location anchor guarded by locality/city checks.
5. **Single-Tenant Address Anchor Channel:** Matches high-confidence single tenants using address tokens and initial character alignment.
6. **Distinctive Brand Token + Street Channel:** Matches rare brand words with street names, disambiguated by token Jaccard ($\ge 0.30$).
7. **Postal Code (PIN/ZIP) + Prefix Channel:** Matches entities sharing postal codes and brand prefixes.

### 4. Indian Metropolitan Hub Conflict Guard
Prevents cross-city false positive collisions between chain stores across major metropolitan hubs (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune).

---

## 📁 Repository Structure

```
Amazon-ML-Challenge/
├── code/
│   └── business_entity_resolution/
│       ├── pipeline_fast_stream.py      # High-speed inverted stream inference engine (CURRENT)
│       ├── experiments/
│       │   ├── log.csv                  # Official benchmark progression log (v01 to v06)
│       │   ├── v05_retrieval_rebound/   # Pipeline v05 experiment code & results
│       │   └── v06_precision_hardening/ # Pipeline v06 experiment code & results
│       ├── src/
│       │   ├── pipeline.py              # Modular entity resolution pipeline
│       │   └── baseline.py              # Initial baseline reference
│       └── requirements.txt             # Environment dependencies
├── dataset/
│   ├── train/                           # Training source files & ground truth
│   └── test/                            # Test source files
├── output/
│   ├── matching_results.tsv             # Leaderboard submission file (1,732,544 rows verified)
│   └── candidate_pairs.tsv              # Candidate blocking file
├── utils/
│   └── validate_submission.py           # Official challenge submission validator
├── .gitignore
└── README.md
```

---

## 🚀 Reproduction & Execution

### 1. Environment Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Generate Submission via Fast Streaming Engine

Run the fast stream pipeline against the test dataset:

```bash
python -u pipeline_fast_stream.py
```

Outputs will be deposited directly in `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### 3. Validate Submission Compliance

Verify that 100% of required test entities are present and format-compliant:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Expected Validation Result:**
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1,732,544
  matching_results.tsv: 1,732,544 rows (17,877 empty, 1,714,667 non-empty).
  candidate_pairs.tsv:  1,732,544 rows (17,877 empty, 1,714,667 non-empty).
PASS — no blocking issues found. Safe to submit.
```