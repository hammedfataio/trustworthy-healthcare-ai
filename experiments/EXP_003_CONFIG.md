# EXP-003 — Experimental Configuration

**Experiment:** EXP-003  
**Title:** Predictive Uncertainty for Error Detection  
**Status:** Implementation complete — execution pending  
**Research stage:** Uncertainty Quantification  
**Execution state:** Not yet executed

---

## 1. Purpose

This document records the experimental configuration for EXP-003 before
execution.

Its purpose is to preserve the experimental conditions under which
predictive uncertainty will be evaluated and to reduce the risk of
post-hoc methodological changes after observing test-set results.

This document does not contain experimental results.

---

## 2. Research Question

> Does predictive uncertainty provide useful information for distinguishing
> incorrect from correct predictions produced by the baseline medical-image
> classifier?

---

## 3. Dataset

**Dataset:** PneumoniaMNIST  
**Source:** MedMNIST v2  
**Task:** Binary medical-image classification

Classes:

- `0` — Normal
- `1` — Pneumonia

Frozen dataset split sizes:

| Split | Samples |
|---|---:|
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |
| Total | 5,856 |

EXP-003 will use the held-out test set for final uncertainty evaluation.

---

## 4. Frozen Baseline Model

EXP-003 does not train a replacement classifier.

It evaluates predictions produced by the frozen EXP-001 baseline CNN.

Checkpoint:

`results/models/experiment_001_baseline_cnn.pt`

The architecture is:

1. Conv2D: 1 → 16 channels
2. ReLU
3. MaxPool2D
4. Conv2D: 16 → 32 channels
5. ReLU
6. MaxPool2D
7. Flatten
8. Linear: 32 × 7 × 7 → 64
9. ReLU
10. Linear: 64 → 1

The final logit is converted to a probability using the sigmoid function.

---

## 5. Frozen EXP-001 Reference Performance

The previously recorded EXP-001 held-out test performance is:

| Metric | Value |
|---|---:|
| Accuracy | 0.884615 |
| AUROC | 0.936993 |
| Sensitivity | 0.9846 |
| Specificity | 0.7179 |
| Precision | 0.8533 |
| F1-score | 0.914286 |
| TN | 168 |
| FP | 66 |
| FN | 6 |
| TP | 384 |

These values are historical reference evidence from EXP-001.

Before accepting EXP-003 results, the loaded checkpoint should reproduce
the relevant baseline prediction behaviour.

---

## 6. Random Seed

The experiment configuration uses:

`SEED = 42`

The baseline model will not be retrained during EXP-003.

The seed is retained for reproducibility of any operations that may depend
on random-number generators.

---

## 7. Batch Size

Test inference batch size:

`64`

Test-set order will not be shuffled.

---

## 8. Classification Threshold

Binary prediction threshold:

`0.5`

Therefore:

- probability `< 0.5` → predicted class `0`
- probability `>= 0.5` → predicted class `1`

The threshold will not be selected using the EXP-003 test results.

---

## 9. Primary Uncertainty Method

The first uncertainty baseline is binary predictive entropy.

For predicted probability `p`:

`H(p) = -p log(p) - (1-p) log(1-p)`

Maximum binary entropy occurs at:

`p = 0.5`

Low entropy occurs when the model probability approaches:

`0` or `1`

---

## 10. Normalized Predictive Entropy

Normalized entropy is defined as:

`H_normalized(p) = H(p) / log(2)`

This places binary predictive entropy approximately within:

`[0, 1]`

where larger values represent greater predictive uncertainty.

---

## 11. Interpretation Boundary

Predictive entropy in EXP-003 is a deterministic uncertainty baseline.

It measures uncertainty implied by the model's output probability.

It should not be interpreted as a complete estimate of epistemic
uncertainty.

A model may be confidently wrong.

For example, an incorrect prediction with probability close to `1.0`
may have very low predictive entropy.

EXP-003 explicitly investigates this limitation.

---

## 12. Error Definition

Prediction correctness is defined as:

- `correct = 1` when predicted label equals true label
- `correct = 0` otherwise

Error target is defined as:

- `error = 0` for correct prediction
- `error = 1` for incorrect prediction

For error-detection evaluation, prediction error is therefore the positive
class.

---

## 13. Primary Evaluation Metrics

The primary EXP-003 error-detection metrics are:

### Error-detection AUROC

Measures how effectively uncertainty ranks incorrect predictions above
correct predictions.

### Error-detection AUPRC

Measures precision-recall performance when prediction error is treated as
the positive class.

AUPRC is particularly relevant because errors are expected to be less
frequent than correct predictions.

### Error prevalence

The proportion of predictions that are incorrect will also be reported.

This provides context for interpreting AUPRC.

---

## 14. Supporting Analysis

EXP-003 will additionally compare uncertainty distributions between:

- correct predictions
- incorrect predictions

Planned descriptive statistics include:

- count
- mean
- median
- standard deviation
- minimum
- 25th percentile
- 75th percentile
- maximum

---

## 15. High-Confidence Errors

The experiment will identify incorrect predictions with:

`confidence >= 0.90`

where:

`confidence = max(p, 1-p)`

These cases are scientifically important because deterministic entropy may
assign low uncertainty to highly confident but incorrect predictions.

---

## 16. Prediction-Level Artifact

Planned output:

`results/tables/experiment_003_prediction_level_results.csv`

Expected fields:

| Field | Meaning |
|---|---|
| sample_id | Test-set sample identifier |
| true_label | Ground-truth class |
| predicted_probability | Probability of pneumonia |
| predicted_label | Binary prediction |
| correct | Correctness indicator |
| error | Error indicator |
| confidence | Maximum class probability |
| predictive_entropy | Binary predictive entropy |
| normalized_predictive_entropy | Entropy normalized by log(2) |

---

## 17. Summary Artifact

Planned output:

`results/tables/experiment_003_uncertainty_summary.csv`

This artifact will summarize uncertainty separately for correct and
incorrect predictions.

---

## 18. Error-Detection Artifact

Planned output:

`results/tables/experiment_003_error_detection_metrics.csv`

Expected metrics include:

- number of samples
- number of errors
- number of correct predictions
- accuracy
- error prevalence
- error-detection AUROC
- error-detection AUPRC

---

## 19. Test-Set Governance

The test set must not be used to:

- select an uncertainty method
- select a classification threshold
- tune model parameters
- tune uncertainty parameters
- select hyperparameters
- repeatedly modify the method until favourable results appear

The test set is reserved for evaluation of the frozen experimental
configuration.

---

## 20. Integrity Checks

Before accepting EXP-003 evidence, execution should verify:

- test set contains 624 samples
- probabilities are finite
- probabilities lie within `[0, 1]`
- entropy values are finite
- entropy values are non-negative
- normalized entropy respects its expected range
- correctness and error indicators are complementary
- required artifact columns exist
- frozen EXP-001 checkpoint loads successfully

---

## 21. Evidence Acceptance

EXP-003 results will not be considered valid merely because the experiment
script executes successfully.

Evidence should only be accepted after:

1. unit tests have been executed successfully;
2. the frozen EXP-001 checkpoint has been loaded;
3. baseline behaviour has been checked against EXP-001;
4. integrity checks have passed;
5. result artifacts have been inspected;
6. calculations have been reviewed for methodological consistency.

---

## 22. Current Status

At the time this configuration was written:

| Component | Status |
|---|---|
| EXP-003 research design | Frozen |
| Predictive entropy implementation | Written |
| Predictive entropy tests | Written |
| Error-detection evaluation | Written |
| Error-detection tests | Written |
| Experiment runner | Written |
| Unit tests executed | Pending |
| EXP-003 executed | Pending |
| Results generated | Pending |
| Results interpreted | Pending |
| EXP-003 final report | Pending |

No EXP-003 result is claimed in this document.

---

## 23. Next Execution Stage

When a local development environment becomes available, the workflow is:

`environment sync → unit tests → checkpoint integrity → EXP-003 execution → artifact inspection → analysis → interpretation → final experiment report`

Until execution occurs, EXP-003 remains an implemented but unevaluated
experiment.
