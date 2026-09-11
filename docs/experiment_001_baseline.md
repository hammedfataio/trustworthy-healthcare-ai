# Experiment 001 — Baseline Classification

**Project:** Trustworthy Healthcare AI  
**Experiment:** 001  
**Status:** ✅ Complete  
**Research Stage:** Baseline Classification

---

## 1. Objective

The objective of this experiment is to establish a baseline machine-learning classification system and evaluate its predictive performance before investigating deeper questions of model trustworthiness.

The experiment addresses the initial question:

> **How well does the baseline model perform, and does strong aggregate performance imply that its individual predictions can be trusted?**

The baseline provides the reference point for subsequent experiments on probability calibration, uncertainty, distribution shift, and robustness.

---

## 2. Dataset

The experiment uses **PneumoniaMNIST**, a medical image classification dataset derived from chest X-ray images.

The task is a binary classification problem involving:

- **Normal**
- **Pneumonia**

The dataset provides a useful controlled environment for investigating model performance and trustworthiness in a healthcare-oriented machine-learning setting.

---

## 3. Research Approach

The baseline experiment follows a standard supervised machine-learning workflow:

1. Load and inspect the dataset
2. Examine class distribution
3. Prepare the data for model training
4. Train the baseline classifier
5. Generate predictions on unseen test data
6. Evaluate conventional classification performance
7. Examine prediction confidence
8. Inspect incorrect predictions
9. Identify potential trustworthiness concerns

The purpose is not only to determine whether the model performs well overall, but also to investigate how its confidence behaves when it makes mistakes.

---

## 4. Evaluation Metrics

The baseline model is evaluated using multiple classification metrics.

These include:

- Accuracy
- Area Under the Receiver Operating Characteristic Curve (AUROC)
- F1-score
- Sensitivity
- Specificity

Using several metrics is important because accuracy alone can hide important differences in model behaviour, particularly when class distributions are unequal or when different types of errors have different consequences.

---

## 5. Baseline Results

The model achieved the following test performance:

| Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| AUROC | 0.9370 |
| F1-score | 91.43% |
| Sensitivity | 98.46% |
| Specificity | 71.79% |

These results indicate strong overall discriminatory performance.

In particular, the high sensitivity suggests that the model identified a large proportion of pneumonia cases in the evaluated test data.

However, the lower specificity indicates that performance was less strong when identifying normal cases.

---

## 6. Beyond Accuracy

The central purpose of this project is not simply to maximise classification performance.

A trustworthy model should also provide meaningful information about the reliability of its predictions.

For this reason, prediction confidence was examined alongside conventional classification metrics.

This produced an important observation.

---

## 7. High-Confidence Failure

During error analysis, the model produced an **incorrect prediction with approximately 99.98% confidence**.

This is significant because the model was not merely incorrect.

It was **extremely confident while being incorrect**.

The observation demonstrates an important distinction:

> **Predictive accuracy and trustworthy confidence are not the same thing.**

A model can achieve strong aggregate performance while still producing individual predictions that may be dangerously overconfident.

---

## 8. Key Finding

Experiment 001 therefore produced two findings.

### Finding 1 — Strong Aggregate Performance

The baseline classifier achieved:

- 88.46% accuracy
- 0.9370 AUROC
- 91.43% F1-score
- 98.46% sensitivity

This establishes that the model has meaningful predictive capability on the evaluated dataset.

### Finding 2 — Confidence Can Still Fail

Despite this performance, an incorrect prediction occurred at approximately **99.98% confidence**.

This creates the central trustworthiness problem investigated by the next stages of the project.

---

## 9. Interpretation

The baseline results demonstrate why conventional classification metrics are necessary but insufficient for evaluating trustworthy AI.

Metrics such as accuracy and AUROC describe performance across a dataset.

They do not necessarily answer questions such as:

- How reliable is an individual prediction?
- Does 90% confidence actually correspond to approximately 90% correctness?
- Can the model recognise difficult examples?
- Can the model indicate when it is uncertain?
- What happens to confidence when the data distribution changes?

The high-confidence error therefore motivates investigation beyond conventional predictive performance.

---

## 10. Research Progression

Experiment 001 establishes the starting point for the research programme:

**Predictive Performance**  
↓  
**Probability Calibration**  
↓  
**Predictive Uncertainty**  
↓  
**Distribution Shift**  
↓  
**Robustness**  
↓  
**Trustworthy AI**

The first transition is from:

> **"How accurate is the model?"**

to:

> **"Can its predicted probabilities be trusted?"**

---

## 11. Limitations

### Dataset Scope

PneumoniaMNIST provides a useful experimental benchmark, but it does not represent the full complexity of real-world clinical environments.

### Aggregate Metrics

The reported metrics summarise performance across the test dataset and do not fully describe prediction-level reliability.

### Confidence Is Not Uncertainty

A high softmax probability should not automatically be interpreted as evidence that a model is certain in a broader statistical or clinical sense.

### Clinical Generalisation

Results from this experimental dataset should not be interpreted as evidence of clinical readiness or real-world diagnostic performance.

---

## 12. Research Implication

The high-confidence failure provides a concrete reason to investigate **probability calibration**.

The next experiment therefore asks:

> **When the model reports a particular confidence level, how closely does that confidence correspond to observed correctness?**

This leads directly to:

### Experiment 002 — Probability Calibration

Experiment 002 investigates calibration using probability-quality measures and post-hoc **temperature scaling**.

[Read Experiment 002 →](experiment_002_calibration.md)

---

## 13. Reproducibility

The code, notebooks, experimental outputs, and project dependencies are maintained within the repository to support reproducibility and future extension.

Related documentation:

- [Project README](../README.md)
- [Research Log](research_log.md)
- [Experiment 002 — Probability Calibration](experiment_002_calibration.md)

---

## Conclusion

Experiment 001 established a strong baseline classifier with **88.46% accuracy and 0.9370 AUROC**.

However, error analysis revealed an incorrect prediction made with approximately **99.98% confidence**.

That observation became the key motivation for the trustworthy-AI research programme developed in this repository.

The experiment therefore shifts the project from asking:

**"Can the model predict?"**

toward the more important question:

**"Can we trust the model when it predicts?"**

---

## Next

➡️ **Experiment 002 — Probability Calibration**
