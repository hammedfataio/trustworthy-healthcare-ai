# EXP-003 — Experimental Configuration

**Experiment:** EXP-003  
**Title:** Predictive Uncertainty for Error Detection  
**Research Stage:** Uncertainty Quantification  
**Status:** Implementation complete — experimental evaluation pending

---

## 1. Purpose

This document records the experimental configuration for EXP-003 before
experimental execution and evaluation.

Its purpose is to preserve the experimental conditions under which predictive
uncertainty will be evaluated and to reduce the risk of post-hoc methodological
changes after observing held-out test results.

This document defines the frozen configuration used to evaluate whether
predictive uncertainty provides useful information about prediction errors
produced by the EXP-001 baseline medical-image classifier.

This document does not contain EXP-003 experimental results.

---

## 2. Research Question

The primary research question is:

> Does predictive uncertainty provide useful information for distinguishing
> incorrect from correct predictions produced by the baseline medical-image
> classifier?

This question extends the previous experiments from predictive performance
and probability calibration toward prediction-level uncertainty assessment.

---

## 3. Research Progression

The current experimental sequence is:

```text
EXP-001
Baseline medical-image classification
        ↓
How well does the model predict?

EXP-002
Probability calibration
        ↓
Do predicted probabilities reflect empirical reliability?

EXP-003
Predictive uncertainty
        ↓
Can uncertainty help identify predictions that are more likely to be wrong?

EXP-004
Selective prediction / referral
        ↓
Can uncertainty support risk-aware prediction handling?

EXP-005
Distribution shift
        ↓
Does uncertainty respond when the data distribution changes?

EXP-006
Robustness
        ↓
How stable are predictions and uncertainty under perturbation?
