# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers. The objective of this challenge is to build a high-performance Machine Learning solution that determines which records across 3 independent sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference source.
- Find all matching records from **Source 2** and **Source 3** for each Source 1 entity.
- A Source 1 entity may match zero (singleton), one, or many records from Source 2 and Source 3.

---

## Evaluation Metric

Submissions are evaluated using the **Macro-Averaged \(F_{0.5}\) Score** — a precision-heavy metric that heavily penalizes false merges (linking different businesses) compared to missed links.

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- Computed per Source 1 entity and macro-averaged across all Source 1 entities in the evaluation set.
- Singletons (Source 1 records with no matches) are included: correctly predicting an empty match list yields 1.0, while false merges yield 0.0.

---

## Dataset Format

All datasets are Tab-Separated Values (`.tsv`).

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains:
1. `entity_id`: Unique record identifier (`S1-`, `S2-`, or `S3-` prefix).
2. `business_name`: Business entity name (with variations, abbreviations, typos, transliterations).
3. `business_address`: Physical address (partial, varied formatting, landmark-based).
4. `country`: Country label (`US`, `India`, and `France` in test set).

### Ground Truth
`train_ground_truth.tsv`:
- `source1_entity_id`: Source 1 entity ID
- `matched_entity_ids`: Comma-separated list of matching entity IDs from Source 2 / Source 3 (or empty).

---

## Repository Structure

```
Amazon-ML-Challenge/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── baseline.py              # End-to-end baseline pipeline
│       ├── README.md                    # Pipeline execution & reproduction guide
│       └── requirements.txt             # Environment dependencies
├── Details/
│   ├── Guidelines.pdf                   # Official challenge guidelines
│   ├── problem statements.pdf           # Detailed problem statement
│   └── Problem Statements Video.mp4     # Overview video
├── student_resource/
│   ├── Documentation_template.md        # Solution methodology documentation template
│   ├── README.md                        # Official student guide
│   ├── code/                            # Resource baseline code
│   └── utils/                           # Validation helper scripts
├── utils/
│   └── validate_submission.py           # Submission validation script
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Installation

Set up a virtual environment and install required dependencies:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Running Baseline Pipeline

```bash
python src/baseline.py
```

### 3. Validating Submission

Before submitting outputs to the leaderboard, validate format and consistency:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Rules & Constraints

- **Model Limits:** Up to 8 Billion parameters with MIT / Apache 2.0 open-source license.
- **Fair Play:** External lookups (commercial APIs, government registries, online search engines) are strictly prohibited.
- **Output Requirements:** Both `matching_results.tsv` and `candidate_pairs.tsv` are required for final submission.