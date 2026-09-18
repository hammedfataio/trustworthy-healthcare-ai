# Trustworthy Healthcare AI

## Uncertainty-Aware, Robust, and Reproducible Machine Learning for Healthcare

This repository documents an evolving research programme investigating **trustworthy artificial intelligence for healthcare**, with an initial focus on medical image classification, probability calibration, uncertainty quantification, selective prediction, distribution shift, and robustness.

The long-term direction is to extend these principles toward **multimodal and generative healthcare AI**, while building an integrated research prototype in which reliability mechanisms are evaluated experimentally before they become system capabilities.

The central principle of the project is:

> **Strong predictive performance alone is not sufficient evidence that an AI system can be trusted.**

---

## Research Motivation

Machine-learning systems can achieve strong predictive performance while still producing unreliable behaviour.

A model may:

- make highly confident incorrect predictions;
- produce poorly calibrated probabilities;
- fail to express increased uncertainty when incorrect;
- deteriorate under distribution shift;
- remain overconfident on unfamiliar inputs; or
- fail unpredictably under perturbation.

These limitations are particularly important in high-stakes domains such as healthcare.

This project therefore investigates not only:

> **Can the model make accurate predictions?**

but also:

> **Can the model recognise when its predictions may be unreliable?**

---

## Central Research Question

The broader research programme investigates:

> **How can uncertainty-aware and robustness-oriented methods improve the reliability of machine-learning systems for healthcare decision support?**

The initial experiments use medical image classification as a controlled foundation before extending toward more complex multimodal and generative AI settings.

---

## Research Objectives

The project aims to:

1. establish a reproducible medical-image classification baseline;
2. evaluate predictive performance beyond accuracy alone;
3. investigate probability calibration and model overconfidence;
4. quantify prediction-level uncertainty;
5. evaluate whether uncertainty can identify model errors;
6. investigate uncertainty-based selective prediction and referral;
7. evaluate reliability under distribution shift;
8. evaluate robustness under controlled perturbation;
9. extend validated principles toward multimodal healthcare AI;
10. investigate trustworthy generative and vision-language AI; and
11. integrate experimentally supported components into a complete research prototype.

---

## Research Progress

| Experiment | Research Question | Status |
|---|---|---|
| EXP-001 — Baseline Classification | How well does the baseline model perform? | ✅ Complete |
| EXP-002 — Probability Calibration | Can we trust the model's predicted probabilities? | ✅ Complete |
| EXP-003 — Uncertainty Quantification | Can uncertainty help identify risky predictions? | 🔬 Next |
| EXP-004 — Selective Prediction | Can uncertainty-based referral reduce risk among retained predictions? | 📋 Planned |
| EXP-005 — Distribution Shift | What happens when the data distribution changes? | 📋 Planned |
| EXP-006 — Robustness | Does model reliability survive challenging conditions? | 📋 Planned |
| Future | Multimodal Healthcare AI | 🔭 Future |
| Future | Trustworthy Generative / Vision-Language AI | 🔭 Future |
| Future | Integrated Trustworthy AI Research Prototype | 🔭 Future |

The roadmap is evidence-driven and may be refined as new experimental findings emerge.

---

# EXP-001 — Baseline Medical Image Classification

## Objective

EXP-001 established a reproducible predictive baseline using **PneumoniaMNIST** from the MedMNIST benchmark collection.

The research question was:

> **How effectively can a baseline convolutional neural network distinguish pneumonia-positive from pneumonia-negative chest X-ray images using the PneumoniaMNIST benchmark?**

---

## Dataset

The experiment uses the predefined PneumoniaMNIST splits.

| Split | Samples |
|---|---:|
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |
| **Total** | **5,856** |

The task is binary classification:

```text
0 → Normal
1 → Pneumonia
```

The training distribution is imbalanced:

```text
Normal:     1,214
Pneumonia:  3,494
```

Therefore, accuracy is not interpreted in isolation.

---

## Baseline Model

The initial model is a compact convolutional neural network consisting of:

```text
Input
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Fully Connected Layer
  ↓
Binary Logit
```

Training configuration includes:

```text
Loss:        BCEWithLogitsLoss
Optimizer:   Adam
Learning Rate: 0.001
Epochs:      10
Batch Size:  64
Seed:        42
```

---

## Baseline Results

Held-out test performance:

| Metric | Result |
|---|---:|
| Accuracy | 0.8846 |
| AUROC | 0.9370 |
| Sensitivity | 0.9846 |
| Specificity | 0.7179 |
| Precision | 0.8533 |
| F1-score | 0.9143 |

Confusion matrix:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 168 | 66 |
| Actual Pneumonia | 6 | 384 |

The model demonstrated strong discrimination and high sensitivity, but substantially lower specificity.

---

## Important Observation

EXP-001 also exposed a central trustworthiness problem.

An incorrectly classified normal image received approximately:

> **99.98% predicted probability of pneumonia**

This demonstrated that strong aggregate predictive performance does not guarantee reliable confidence for individual predictions.

That observation motivated EXP-002.

---

# EXP-002 — Probability Calibration

## Research Question

> **How well calibrated are the probability estimates produced by the baseline CNN, and does the model exhibit overconfidence when making incorrect predictions?**

The experiment investigated:

- reliability diagrams;
- Expected Calibration Error;
- Brier score;
- confidence behaviour;
- high-confidence errors; and
- temperature scaling.

---

## Temperature Scaling

Temperature scaling was fitted using the **validation set**, not the held-out test set.

The final validation-fitted temperature was:

```text
T = 1.007948
```

Validation negative log-likelihood changed from:

```text
Before: 0.097432
After:  0.097427
```

The improvement was negligible.

The appropriate conclusion is therefore limited to the current experimental setting:

> **Validation-fitted global temperature scaling produced negligible improvement for the baseline model under the evaluated conditions.**

This does **not** establish that temperature scaling is generally ineffective for medical AI.

---

## Why EXP-002 Matters

Calibration is a population-level property.

Even a reasonably calibrated model can still make individual high-confidence errors.

This creates the next research question:

> **Can prediction-level uncertainty provide useful information about model failure that confidence alone cannot?**

That question defines EXP-003.

---

# Research Direction

The experimental programme progresses through increasingly demanding questions:

```text
Predictive Performance
        ↓
Probability Calibration
        ↓
Uncertainty Quantification
        ↓
Selective Prediction
        ↓
Distribution Shift
        ↓
Robustness
        ↓
Multimodal Healthcare AI
        ↓
Trustworthy Generative / Vision-Language AI
        ↓
Integrated Trustworthy AI Research Prototype
```

Each stage should generate evidence before the corresponding capability is integrated into the wider system.

---

# EXP-003 — Uncertainty Quantification

**Status: Next**

EXP-003 will investigate whether uncertainty estimates provide useful information about prediction failure.

The initial research question is:

> **Can uncertainty distinguish incorrect predictions from correct predictions in the baseline medical-image classification setting?**

Planned evaluation includes:

- predictive entropy;
- uncertainty distributions for correct and incorrect predictions;
- error-detection AUROC;
- error-detection AUPRC;
- analysis of low-uncertainty errors;
- comparison with an explicit uncertainty method where justified; and
- preparation for selective-prediction evaluation.

The existing EXP-001 CNN contains no dropout layers.

Therefore, MC Dropout cannot simply be applied to the existing checkpoint without modifying the architecture and retraining.

The uncertainty methodology will be defined explicitly before implementation.

---

# From Experiments to a Complete System

This repository is not intended to remain a collection of disconnected notebooks.

Validated research components will progressively contribute to a broader system.

The target research architecture is:

```text
Healthcare Input
      ↓
Data Processing
      ↓
Predictive / Multimodal Model
      ↓
Prediction or Generated Output
      ↓
Trustworthiness Layer
      ├── Calibration
      ├── Uncertainty
      ├── Distribution-Shift Evaluation
      └── Robustness
      ↓
Risk Assessment
      ↓
Lower-Risk Output ──────────────► Research Interface
      │
      └── Higher Uncertainty
                ↓
          Flag / Refer
                ↓
          Human Review
                ↓
        Audit and Monitoring
```

This is a **research prototype direction**, not a clinically validated diagnostic system.

---

# Research, Engineering, and Publication Tracks

The project develops through three connected tracks.

## Research Track

```text
Research Question
      ↓
Hypothesis
      ↓
Experiment
      ↓
Evaluation
      ↓
Evidence
```

## System Engineering Track

```text
Validated Evidence
      ↓
Reusable Component
      ↓
Tested Pipeline
      ↓
Integrated Research System
```

## Publication Track

```text
Experimental Evidence
      ↓
Analysis
      ↓
Figures and Tables
      ↓
Research Contribution
      ↓
Manuscript / Preprint
```

The tracks remain connected through experimental evidence.

---

# Evaluation Philosophy

The project deliberately separates different dimensions of model behaviour.

```text
Accuracy ≠ Calibration

Calibration ≠ Uncertainty

Uncertainty ≠ Robustness

Strong AUROC ≠ Trustworthy AI
```

Evaluation therefore considers complementary dimensions including:

- predictive performance;
- discrimination;
- calibration;
- uncertainty;
- error detection;
- selective prediction;
- distribution shift;
- robustness; and
- eventually multimodal and generative reliability.

Negative or negligible results are retained when the experimental procedure is valid.

---

# Reproducibility

The project uses:

```text
Python 3.11
uv
PyTorch
MedMNIST
scikit-learn
Jupyter
```

Environment information is maintained through:

```text
.python-version
pyproject.toml
uv.lock
```

The project aims to preserve traceability from:

```text
Research Question
        ↓
Code
        ↓
Environment
        ↓
Dataset
        ↓
Model
        ↓
Evaluation
        ↓
Artifact
        ↓
Interpretation
```

---

# Repository Structure

```text
trustworthy-healthcare-ai/
│
├── docs/
│   ├── RESEARCH_PLAN.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENTS.md
│   ├── EVALUATION.md
│   ├── REPRODUCIBILITY.md
│   ├── PUBLICATION_PLAN.md
│   ├── research_log.md
│   ├── experiment_001_baseline.md
│   └── experiment_002_calibration.md
│
├── experiments/
│
├── notebooks/
│   ├── 01_baseline_medical_imaging.ipynb
│   └── 02_confidence_calibration.ipynb
│
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── uncertainty/
│
├── tests/
│
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

The structure will evolve as experimentally validated components become reusable research software.

---

# Research Documentation

The documentation is organised so that the project can be followed from the central research problem through methodology, experimentation, evaluation, reproducibility, and publication planning.

## Research Framework

- [Research Plan](docs/RESEARCH_PLAN.md) — research problem, aims, questions, objectives, scope, and roadmap.
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md) — progression from experimental evidence toward an integrated trustworthy AI research prototype.
- [Methodology](docs/METHODOLOGY.md) — datasets, experimental design, modelling procedures, controls, and research methodology.
- [Experiment Registry](docs/EXPERIMENTS.md) — master registry of planned, active, and completed experiments.
- [Evaluation Framework](docs/EVALUATION.md) — evaluation dimensions, metrics, interpretation, and limitations.
- [Reproducibility Framework](docs/REPRODUCIBILITY.md) — environment, experiment traceability, artifacts, and reproduction standards.
- [Publication Plan](docs/PUBLICATION_PLAN.md) — pathway from experimental evidence toward a potential research manuscript.

## Experimental Evidence

- [EXP-001 — Baseline Medical Image Classification](docs/experiment_001_baseline.md)
- [EXP-002 — Probability Calibration](docs/experiment_002_calibration.md)

## Research Progress

- [Research Log](docs/research_log.md) — chronological record of experiments, findings, limitations, and research decisions.

Future experiment reports will be added after the corresponding experiments are completed.

---

# Publication Direction

The current provisional publication direction is:

> **Beyond Accuracy: Evaluating Uncertainty and Robustness in Deep Learning for Medical Image Classification Under Distribution Shift**

This is a working direction rather than a predetermined paper title.

The final research contribution will be determined by the experimental evidence produced through EXP-003 and subsequent experiments.

The project follows the principle:

> **The evidence determines the claim — the desired claim does not determine which evidence is reported.**

---

# Current Limitations

The project is currently at an early research stage.

Current experiments do not establish:

- clinical effectiveness;
- clinical safety;
- generalisation across hospitals;
- generalisation across patient populations;
- robustness to real-world acquisition differences;
- reliable uncertainty quantification;
- effective uncertainty-based referral;
- multimodal reliability;
- generative AI reliability; or
- suitability for clinical deployment.

These are research questions rather than assumed capabilities.

---

# Current Position

The research programme currently stands at:

```text
EXP-001
Baseline Classification
        │
        └── COMPLETE
                ↓
EXP-002
Probability Calibration
        │
        └── COMPLETE
                ↓
EXP-003
Uncertainty Quantification
        │
        └── NEXT
                ↓
EXP-004
Selective Prediction
                ↓
EXP-005
Distribution Shift
                ↓
EXP-006
Robustness
                ↓
Multimodal / Generative Extension
                ↓
Integrated Trustworthy AI Research Prototype
                ↓
Research Publication
```

The immediate research priority is:

> **EXP-003 — determine whether uncertainty provides useful information about model failure.**

---

## Research Status

**Active Research**

Current stage:

**EXP-003 — Uncertainty Quantification and Error Detection**

Completed evidence:

**EXP-001 + EXP-002**

Next objective:

**Move from probability confidence toward explicit evaluation of prediction-level uncertainty.**
