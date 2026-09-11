# Experiment 002 — Probability Calibration

**Project:** Trustworthy Healthcare AI  
**Experiment:** 002  
**Status:** ✅ Complete  
**Research Stage:** Probability Calibration

---

## 1. Objective

The objective of this experiment is to investigate whether the confidence scores produced by the baseline model accurately reflect the likelihood that its predictions are correct.

Experiment 001 demonstrated that strong classification performance does not necessarily guarantee trustworthy individual predictions. In particular, an incorrect prediction was observed with approximately **99.98% confidence**.

This motivates the research question:

> **When the model reports high confidence, how well does that confidence correspond to its actual probability of being correct?**

---

## 2. Research Hypothesis

The baseline model may exhibit a mismatch between predictive confidence and observed correctness.

Post-hoc probability calibration may reduce this mismatch without changing the model's underlying class predictions.

---

## 3. Methodology

The experiment evaluates model confidence before and after calibration.

The workflow consists of:

1. Obtaining predictions from the trained baseline model
2. Retaining model outputs required for calibration
3. Evaluating baseline classification performance
4. Evaluating probability quality
5. Applying **temperature scaling**
6. Evaluating the calibrated probabilities
7. Comparing results before and after calibration

### Temperature Scaling

Temperature scaling is a post-hoc calibration method that learns a single temperature parameter, **T**.

The temperature rescales model logits before probabilities are calculated:

**Calibrated logits = Original logits / T**

In general:

- `T > 1` softens predicted probabilities
- `T < 1` sharpens predicted probabilities
- `T ≈ 1` indicates little global adjustment

The underlying classifier parameters are not retrained.

---

## 4. Evaluation

Calibration is considered separately from classification performance.

The experiment considers:

- Negative Log-Likelihood (NLL)
- Brier Score
- Expected Calibration Error (ECE)
- Reliability analysis
- Accuracy
- AUROC
- F1-score

Strong classification performance does not automatically imply reliable probability estimates.

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

### Learned Temperature

**Learned temperature: `1.007948`**

The learned temperature is very close to `1.0`, indicating that only a small global adjustment to the model logits was identified.

### Validation Negative Log-Likelihood

| Metric | Before | After |
|---|---:|---:|
| Validation NLL | 0.097432 | 0.097427 |

The change in validation NLL is very small.

---

## 6. Key Finding

Temperature scaling produced **negligible change** in validation NLL.

The learned temperature of approximately `1.008` suggests that the calibration procedure identified little need for global rescaling of the model's logits on the calibration data.

Classification performance remained unchanged:

- Accuracy: `0.8846 → 0.8846`
- AUROC: `0.9370 → 0.9370`
- F1-score: `0.9143 → 0.9143`
- Changed class predictions: `0`

This is expected because temperature scaling changes probability confidence rather than the ranking of class scores.

---

## 7. Interpretation

The limited change after temperature scaling is an informative result.

It suggests that the high-confidence failure observed during the baseline experiment cannot be explained simply by a large global calibration problem that temperature scaling can correct.

More importantly, good aggregate calibration does not guarantee that every individual prediction is trustworthy.

A model may appear reasonably calibrated across a dataset while still making high-confidence errors on particular examples.

This moves the research toward a deeper question:

> **Can the model recognise when an individual prediction is uncertain or potentially unreliable?**

---

## 8. Why This Matters

For healthcare-oriented AI, predictive accuracy alone is insufficient.

The research progression is:

**Predictive Performance → Probability Calibration → Predictive Uncertainty → Distribution Shift → Robustness → Trustworthy AI**

Experiment 002 investigates the probability-calibration stage of this progression.

---

## 9. Limitations

**Global calibration:** Temperature scaling learns a single global parameter and may therefore fail to address example-specific or subgroup-specific uncertainty.

**Distribution dependence:** Calibration observed on one validation distribution does not guarantee that probabilities remain reliable when the underlying data distribution changes.

**Aggregate vs individual reliability:** Good average calibration does not mean that every high-confidence prediction is trustworthy.

**Predictive uncertainty:** This experiment does not yet explicitly quantify different sources of predictive uncertainty.

These limitations motivate the next experimental stage.

---

## 10. Next Experiment

### Experiment 003 — Uncertainty Quantification

The next experiment will investigate whether uncertainty estimates can help identify predictions that deserve greater caution.

Planned areas of investigation include:

- Predictive uncertainty
- Uncertainty for correct versus incorrect predictions
- Confidence versus uncertainty
- High-confidence failure cases
- Uncertainty distributions
- Identification of potentially risky predictions

The next research question is:

> **Can uncertainty provide information about model reliability that confidence and calibration alone cannot provide?**

---

## 11. Reproducibility

Implementation code, notebooks and experimental outputs associated with this work are maintained within the repository to support reproducibility and future extension.

Related documentation:

- [Project README](../README.md)
- [Research Log](research_log.md)

---

## Conclusion

Experiment 002 demonstrates an important distinction between **classification performance, probability calibration and predictive uncertainty**.

Temperature scaling learned a temperature of approximately `1.008` and produced only a very small reduction in validation NLL while leaving classification predictions unchanged.

Rather than suggesting that model trustworthiness has been solved, this result motivates the next stage of the research:

**prediction-level uncertainty quantification.**
