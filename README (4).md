# Data-Driven Classification of U.S. Unemployment Regimes

Classifying U.S. unemployment regimes using macroeconomic indicators from the Federal Reserve's **FRED-MD** database, via feature selection and ensemble learning.

**Best result:** ReliefF + Random Forest — **94.38% accuracy**, **Kappa = 0.925**

---

## Overview

This project applies machine learning to classify the U.S. unemployment rate into four categorical regimes (Low / Moderate-Low / Moderate-High / High) using monthly macroeconomic indicators. Five feature-selection methods were paired with four classifiers — 20 model combinations in total — to identify which economic signals and algorithms best capture labor-market dynamics.

The work builds on McCracken & Ng's (2016) FRED-MD database, extending it into a modern, interpretable machine learning framework for detecting cyclical transitions in the labor market.

## Dataset

- **Source:** `2025-08-MD.csv` — FRED-MD, Federal Reserve Bank of St. Louis
- **Coverage:** 800 monthly observations, January 1959 – August 2025
- **Features:** 127 macroeconomic indicators (employment, industrial production, money supply, interest rates, consumer spending, etc.)
- **Target:** `UNRATE` (unemployment rate), discretized into four equal-frequency bins:
  - `(-∞, 4.65]` — Low
  - `(4.65, 5.65]` — Moderate-Low
  - `(5.65, 6.95]` — Moderate-High
  - `(6.95, ∞)` — High

## Methodology

1. **Preprocessing** — Missing values handled via WEKA's `ReplaceMissingValues`; no scaling/standardization applied to preserve interpretability.
2. **Discretization** — Equal-frequency binning (4 bins) chosen over equal-width binning after the latter produced skewed class distributions.
3. **Train/test split** — Stratified 80/20 split (chronological splitting was tried first but caused severe distribution shift and near-collapse of model performance).
4. **Feature selection** — Four algorithms plus a manually curated subset:
   - Information Gain
   - Gain Ratio
   - Symmetrical Uncertainty
   - ReliefF
   - Manual subset (domain-informed)
5. **Classifiers** — Random Forest, J48 (C4.5), Logistic Regression, OneR
6. **Evaluation** — Accuracy, Kappa, precision/recall/F1, ROC/PRC area, and confusion matrices on a held-out 160-instance stratified test set, with 10-fold cross-validation on training data.

## Results

| Attribute Selector | Best Model | Accuracy | Kappa | Notes |
|---|---|---|---|---|
| Information Gain | J48 | 91.88% | 0.892 | Stable, minor mid-bin confusion |
| Gain Ratio | Random Forest | 92.50% | 0.900 | Clean class separation |
| **ReliefF** | **Random Forest** | **94.38%** | **0.925** | Best recall, minimal errors |
| Symmetrical Uncertainty | Random Forest | 93.75% | 0.918 | High stability |
| Manual Subset | Random Forest | 92.50% | 0.900 | Interpretable, competitive accuracy |

**Gold-star model — ReliefF + Random Forest:** Precision ≈ 0.95, Recall ≈ 0.94, Specificity ≈ 0.96, F1 ≈ 0.94, ROC ≈ 0.99. Extreme unemployment bins were classified almost perfectly; errors were concentrated in adjacent middle bins (economically expected "off-by-one" confusion).

## Key Takeaways

- **Random Forest** dominated across feature-selection methods, best capturing nonlinear interactions among labor, price, and monetary indicators.
- **ReliefF**'s local-neighbor ranking surfaced short-term labor/financial indicators (e.g., `UEMPMPAN`, `HWIURATIO`, `BAAFFM`, `T10YFFM`) that Information Gain/Gain Ratio/Symmetrical Uncertainty missed — and this locality paired especially well with Random Forest's ensemble averaging.
- A **manually curated feature subset**, built from domain knowledge, performed nearly as well as automated selection — validating that economic intuition can rival algorithmic ranking.
- Middle unemployment bins (moderate regimes) were consistently the hardest to classify; extreme bins (very low/high unemployment) were the easiest.

## Tools

- **WEKA** — attribute selection, classification, evaluation
- Feature selection: `InfoGainAttributeEval`, `GainRatioAttributeEval`, `SymmetricalUncertAttributeEval`, `ReliefFAttributeEval`
- Classifiers: `weka.classifiers.trees.RandomForest`, `weka.classifiers.trees.J48`, `weka.classifiers.functions.Logistic`, `weka.classifiers.rules.OneR`

## Limitations

- Macroeconomic relationships (e.g., unemployment–inflation links) can shift under new policy regimes, limiting long-term generalizability.
- Discretizing a continuous variable (UNRATE) loses granularity — small but meaningful differences (e.g., 5.9% vs. 6.1%) become categorical.
- Macroeconomic data is autocorrelated over time; Random Forest and J48 don't explicitly model this temporal dependence.
- Evaluation is a static snapshot; real-world deployment would need continuous retraining as new FRED data arrive.

## Future Work

- Time-series-aware models (LSTM, Temporal Random Forests) to capture sequential dependence
- Dynamic/rolling feature re-selection as new data arrive
- Hybrid regression-classification approaches to retain continuous unemployment information
- Incorporating external variables (consumer confidence, fiscal policy indices)
- Evolving into a real-time, continuously retrained economic monitoring tool

## Authors

Navya Arora, Esha Madamalla

## References

McCracken, M. W., & Ng, S. (2016). *FRED-MD: A Monthly Database for Macroeconomic Research.* Federal Reserve Bank of St. Louis.
