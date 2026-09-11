# Experiment 002 — Probability Calibration

**Project:** Trustworthy Healthcare AI  
**Experiment:** 002  
**Focus:** Probability Calibration  
**Status:** ✅ Complete

---

## Objective

The objective of this experiment is to investigate whether the confidence
scores produced by the baseline healthcare classification model accurately
reflect the likelihood that its predictions are correct.

Experiment 001 demonstrated that strong classification performance does not
necessarily guarantee trustworthy individual predictions. In particular, a
misclassified example was observed with approximately **99.98% confidence**.

This motivates the question:

> **When the model reports high confidence, how well does that confidence
> correspond to its actual probability of being correct?**

---

## Research Hypothesis

The baseline model may exhibit a mismatch between predictive confidence and
observed correctness.

Post-hoc probability calibration may reduce this mismatch without changing
the model's underlying class predictions.

---

## Method

The experiment evaluates model confidence before and after calibration.

The workflow consists of:

1. Obtaining predictions from the trained baseline model
2. Retaining model outputs required for calibration
3. Evaluating baseline classification performance
4. Measuring probability quality
5. Applying **temperature scaling**
6. Re-evaluating the calibrated probabilities
7. Comparing performance before and after calibration

### Temperature Scaling

Temperature scaling is a post-hoc calibration technique that learns a single
temperature parameter **T**.

The temperature rescales model logits before probabilities are calculated,
without retraining the underlying classifier.

Conceptually:

```text
Calibrated logits = Original logits / T
