# Experiment 002 — Probability Calibration

**Project:** Trustworthy Healthcare AI  
**Experiment:** 002  
**Status:** ✅ Complete  
**Research Stage:** Probability Calibration

---

## 1. Objective

The objective of this experiment is to investigate whether the confidence
scores produced by the baseline model accurately reflect the likelihood that
its predictions are correct.

Experiment 001 demonstrated that strong classification performance does not
necessarily guarantee trustworthy individual predictions.

In particular, an incorrect prediction was observed with approximately
**99.98% confidence**.

This motivates the research question:

> **When the model reports high confidence, how well does that confidence
> correspond to its actual probability of being correct?**

---

## 2. Research Hypothesis

The baseline model may exhibit a mismatch between predictive confidence and
observed correctness.

Post-hoc probability calibration may reduce this mismatch without changing
the model's underlying class predictions.

---

## 3. Methodology

The experiment evaluates model confidence before and after calibration.

The workflow consists of:

1. obtaining predictions from the trained baseline model;
2. retaining the model outputs required for calibration;
3. evaluating baseline classification performance;
4. evaluating probability quality;
5. applying **temperature scaling**;
6. evaluating the calibrated probabilities;
7. comparing results before and after calibration.

### Temperature Scaling

Temperature scaling is a post-hoc calibration method that learns a single
temperature parameter, **T**.

The temperature rescales model logits before probabilities are calculated:

```text
Calibrated logits = Original logits / T
Learned temperature: 1.007948
Validation NLL before: 0.097432
Validation NLL after:  0.097427
Accuracy: 0.8846 → 0.8846
AUROC:    0.9370 → 0.9370
F1-score: 0.9143 → 0.9143

Changed class predictions: 0
Predictive Performance
        ↓
Probability Calibration
        ↓
Predictive Uncertainty
        ↓
Distribution Shift
        ↓
Robustness
        ↓
Trustworthy AI

That's **everything for this file**. Don't add the README text or the research-log text to it.

Then commit it with:

`Document Experiment 002 calibration methodology and findings`

After you commit, **stop there**. Send me **DONE**.

Then we'll handle the files separately in this order:

`README.md` → `docs/research_log.md` → check whether we need `experiment_001_baseline.md` → then create `experiment_003_uncertainty.md` when Experiment 003 actually starts.

This way we clean the repository instead of accidentally duplicating information across all the `.md` files. 
