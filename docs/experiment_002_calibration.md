# Experiment 002 — Probability Calibration

**Project:** Trustworthy Healthcare AI  
**Experiment:** 002  
**Focus:** Probability Calibration  
**Status:** Complete

---

## 1. Objective

The objective of this experiment is to investigate whether the confidence
scores produced by the baseline healthcare classification model accurately
reflect the likelihood that its predictions are correct.

Experiment 001 demonstrated that strong classification performance does not
necessarily guarantee trustworthy individual predictions. In particular, a
misclassified example was observed with approximately **99.98% confidence**.

This motivates a more specific question:

> **When the model reports high confidence, how well does that confidence
> correspond to its actual probability of being correct?**

---

## 2. Research Hypothesis

The baseline model may exhibit a mismatch between predictive confidence and
observed correctness.

Post-hoc probability calibration may reduce this mismatch without changing
the model's underlying class predictions.

---

## 3. Method

The experiment evaluates model confidence before and after calibration.

The calibration workflow includes:

1. obtaining predictions from the trained baseline model;
2. retaining the model outputs required for calibration;
3. evaluating classification performance;
4. measuring probability quality;
5. applying **temperature scaling**;
6. evaluating the calibrated probabilities;
7. comparing results before and after calibration.

Temperature scaling learns a single temperature parameter **T** that adjusts
the scale of the model logits before probabilities are calculated.

The underlying model parameters remain unchanged.

---

## 4. Evaluation

Calibration is considered separately from classification accuracy.

The analysis uses measures including:

- **Negative Log-Likelihood (NLL)**
- **Brier Score**
- **Expected Calibration Error (ECE)**
- reliability analysis
- conventional classification metrics for comparison

This distinction is important because a model can classify examples
correctly while still producing poorly calibrated confidence estimates.

---

## 5. Results

### Classification Performance

| Metric | Before Calibration | After Temperature Scaling |
|---|---:|---:|
| Accuracy | 0.8846 | 0.8846 |
| AUROC | 0.9370 | 0.9370 |
| F1-score | 0.9143 | 0.9143 |

**Changed class predictions:** `0`

Temperature scaling therefore did not alter the predicted classes.

---

### Learned Temperature

```text
T = 1.007948
