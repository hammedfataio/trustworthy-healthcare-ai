# Trustworthy Healthcare AI

> **When a medical AI model reports 99% confidence, how trustworthy is that confidence?**

A research portfolio investigating **trustworthy artificial intelligence for healthcare**, with an emphasis on model reliability, probability calibration, uncertainty, robustness, and reproducible machine-learning evaluation.

The project explores an important question for high-stakes AI:

> **Can we identify when a machine-learning model should — and should not — be trusted?**

---

## 🔬 Research Motivation

Predictive performance alone is not sufficient for high-stakes applications.

A model may achieve strong accuracy or AUROC while still producing predictions that are:

- overconfident,
- poorly calibrated,
- unreliable on difficult cases, or
- vulnerable to changes in the underlying data distribution.

In healthcare-oriented AI, these limitations are particularly important because model confidence may influence how predictions are interpreted.

This project therefore investigates AI systems beyond conventional predictive performance, focusing on **confidence, calibration, uncertainty and reliability**.

---

## 🧪 Experimental Roadmap

The portfolio is being developed incrementally through reproducible experiments.

### Experiment 001 — Baseline Classification & Confidence Analysis ✅

Established the baseline classification pipeline and examined model confidence alongside conventional predictive performance.

Observed performance:

| Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| AUROC | 0.9370 |
| F1-score | 0.9143 |
| Sensitivity | 98.46% |
| Specificity | 71.79% |

A particularly important observation was the presence of an **incorrect prediction made with high confidence**.

This demonstrates why accuracy alone does not fully describe whether a model's predictions can be trusted.

---

### Experiment 002 — Probability Calibration ✅

Investigated whether predicted probabilities accurately reflect model confidence.

**Temperature scaling** was applied as a post-hoc calibration method.

The learned temperature was:

```text
T = 1.007948
