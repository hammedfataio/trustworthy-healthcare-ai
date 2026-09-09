# Research Log: Trustworthy AI for Healthcare

## From Accurate Predictions to Trustworthy Decisions

### Research Question

**When a medical AI model reports 90% or 99% confidence, how trustworthy is that confidence?**

Machine learning models used in healthcare are often evaluated primarily
through predictive performance metrics such as accuracy, F1-score, and AUROC.

However, high predictive performance does not necessarily mean that a model
understands when it may be wrong.

This distinction is particularly important in healthcare.

A model that makes an incorrect prediction with low confidence may signal
uncertainty and allow the case to be referred for human review. In contrast,
a model that makes an incorrect prediction with extremely high confidence
could create a much more serious reliability problem.

My current research exploration therefore focuses on a broader question:

> **How can we develop medical AI systems that are not only accurate, but also
> calibrated, uncertainty-aware, and robust when encountering unfamiliar or
> shifted clinical data?**

---

## Experiment 001: Establishing the Baseline

As the first stage of this investigation, I developed a baseline convolutional
neural network using the **PneumoniaMNIST** medical imaging benchmark.

The purpose of this experiment was not to optimise benchmark performance.
Instead, the objective was to establish a reproducible reference model whose
behaviour can later be examined under calibration, uncertainty, and
distribution-shift experiments.

The model was trained to distinguish between:

- Normal chest X-ray images
- Pneumonia-positive chest X-ray images

The training dataset contained 4,708 images and exhibited substantial class
imbalance:

| Class | Samples | Percentage |
|---|---:|---:|
| Normal | 1,214 | 25.79% |
| Pneumonia | 3,494 | 74.21% |

This immediately raised an important evaluation issue.

A classifier predicting the majority pneumonia class for every training
sample could achieve approximately **74.21% accuracy** without learning a
clinically useful decision boundary.

Therefore, accuracy alone would not provide an adequate assessment of the
model.

---

## Baseline Results

Evaluation was performed on the held-out test set, which was not used during
model training or validation.

| Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| AUROC | 0.9370 |
| Sensitivity | 98.46% |
| Specificity | 71.79% |
| Precision | 85.33% |
| F1-score | 91.43% |

The model correctly identified **384 of 390 pneumonia cases**, resulting in
a sensitivity of 98.46%.

However, specificity was considerably lower at 71.79%. Of the normal cases,
66 were incorrectly classified as pneumonia.

This difference between sensitivity and specificity demonstrates why
aggregate performance metrics should be interpreted carefully in medical AI.

---

## An Observation That Changed the Next Experiment

One result was particularly interesting.

During inspection of the model's predicted probabilities, I observed an
incorrect prediction for which the model assigned approximately **99.98%
probability to pneumonia**.

The true label was normal.

In other words, the model was not simply wrong.

**It was extremely confident while being wrong.**

This observation motivates the next stage of the research.

If a model reports:

> "Pneumonia probability: 99.98%"

what does that probability actually mean?

Can clinicians—or downstream clinical decision-support systems—interpret
99.98% as a reliable estimate of confidence?

High accuracy does not automatically answer this question.

---

# Experiment 002: Can We Trust Model Confidence?

The next experiment will investigate **probability calibration**.

A well-calibrated model should exhibit a meaningful relationship between its
reported confidence and its empirical correctness.

For example, among predictions made with approximately 90% confidence, we
would ideally expect the model to be correct approximately 90% of the time.

This will be investigated using:

- Reliability diagrams
- Expected Calibration Error (ECE)
- Brier score
- Confidence distributions
- Analysis of high-confidence errors

The goal is to determine whether the baseline CNN's probability estimates
represent meaningful confidence or whether the network exhibits systematic
overconfidence.

---

## Where This Research Is Going

Calibration is only one component of trustworthy medical AI.

The longer-term research direction will progressively investigate:

**Medical Image Classification**
↓
**Probability Calibration**
↓
**Uncertainty Quantification**
↓
**Distribution Shift**
↓
**Robustness**
↓
**Multimodal Clinical AI**
↓
**Trustworthy Generative AI for Healthcare**

An important future question is whether uncertainty estimates remain reliable
when models encounter data that differ from their training distribution.

Ultimately, I am interested in systems capable of recognising situations in
which their predictions should not be trusted and where **human review should
be prioritised**.

---

## Research Philosophy

For high-stakes AI systems, the objective should not simply be:

> **Can the model make the correct prediction?**

It should also include:

> **Does the model know when its prediction may be unreliable?**

This repository documents my ongoing experiments exploring that question.

The work is currently experimental and intended for research and educational
purposes. It is not a medical device and should not be used for clinical
diagnosis.