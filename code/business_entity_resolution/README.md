# Amazon ML Challenge 2026: Business Entity Resolution

## Production Solution — High-Precision Inverted Streaming Engine (v11)

### 1. Overview
In large-scale commercial platforms, entity records arrive from noisy, independent sources lacking shared identifiers. This package implements a high-performance, precision-oriented Machine Learning solution to resolve business entities from **Source 1** (~1.73M records) against **Source 2** (~4.89M records) and **Source 3** (~5.08M records).

The primary evaluation metric is **Macro-Averaged $F_{0.5}$**, which weights precision twice as heavily as recall:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

---

### 2. Architecture & Key Innovations (v11)

1. **Inverted Streaming Inference Engine (`pipeline_fast_stream.py`):**
   - Inverts the indexing direction: indexes Source 1 (~1.8 GB RAM) into compact hash tables, then streams Source 2 and Source 3 sequentially line-by-line.
   - Prevents RAM thrashing and disk swapping, completing inference across **11.7 million records in ~18–20 minutes**.

2. **Strict Geographic Gating (State Veto):**
   - Enforces regional consistency across 30 Indian states, 50 US states, and French departments.
   - Cross-state candidate collisions on common company names (`Om Constructions`, `Vision Partners`, `Red Perfect Trading`) are strictly vetoed.

3. **Physical Anchor & Address Overlap Proof:**
   - Matches require verifiable location identity: identical building number, matching postal code, or overlapping street/locality tokens.
   - Zero address overlap between records with addresses results in immediate disqualification, eliminating over 5.58 million spurious false positives.

4. **Macro $F_{0.5}$ Precision Filtering:**
   - Only high-confidence matches ($\text{score} \ge 85$) are admitted to `matching_results.tsv`.
   - Per-source matches are capped at top 2 (top 3 for $\text{score} \ge 95$), bringing average predictions per entity to **2.48** (aligning with ground truth's 3.46).

5. **Universal Algorithmic Indic Transliteration:**
   - Direct phonetic mapping of Indic Brahmic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati) to Latin bases.

---

### 3. Environment & Dependencies
- Python 3.8+ (Tested on Python 3.10 / 3.11)
- Install requirements:
  ```bash
  pip install -r requirements.txt
  ```

---

### 4. Reproduction & Inference

Run the production streaming pipeline:

```bash
python -u pipeline_fast_stream.py
```

Outputs are automatically saved to:
- `../../output/matching_results.tsv` (1,732,544 rows, 4.29M matches)
- `../../output/candidate_pairs.tsv` (1,732,544 rows)

Validate submission compliance:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
