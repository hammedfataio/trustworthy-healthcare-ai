# Trustworthy Healthcare AI

> **When a healthcare AI model reports 99% confidence, how trustworthy is that confidence?**

A reproducible research and engineering project investigating **uncertainty-aware, calibrated, robust, and risk-aware artificial intelligence for healthcare**.

The project progresses from controlled medical image classification experiments toward a broader trustworthy healthcare AI system capable of evaluating not only **what a model predicts**, but also **when its predictions may be unreliable**.

---

## Research Motivation

Strong predictive performance does not necessarily imply trustworthy AI.

A healthcare model may achieve high accuracy or AUROC while still:

* assigning high confidence to incorrect predictions;
* producing poorly calibrated probabilities;
* failing on unfamiliar or shifted data;
* providing misleading measures of certainty; or
* failing to identify cases that require additional review.

This project investigates these limitations systematically.

Rather than treating model accuracy as the final objective, the research studies the relationship between:

**Prediction → Confidence → Uncertainty → Reliability → Risk**

The longer-term goal is to translate findings from these experiments into a reproducible research system for trustworthy multimodal and generative healthcare AI.

---

## Central Research Question

> **How can uncertainty-aware and robust machine-learning methods improve the reliability of healthcare AI systems, particularly when models encounter difficult cases or changes in data distribution?**

A longer-term question guides the multimodal and generative stages of the project:

> **How can healthcare AI systems integrate multiple clinical data modalities while providing meaningful uncertainty and reliability information to support safer decision-making?**

---

## Research Programme

The project follows an evidence-driven progression.

```text
Medical Data
     │
     ▼
Predictive Model
     │
     ▼
Baseline Performance
     │
     ├──► Probability Calibration
     │
     ├──► Uncertainty Quantification
     │
     ├──► Selective Prediction / Referral
     │
     ├──► Distribution Shift
     │
     └──► Robustness
              │
              ▼
       Multimodal Healthcare AI
              │
              ▼
       Trustworthy Generative AI
              │
              ▼
    Integrated Research Prototype
```

Future experiments are planned in advance, but their conclusions are not.

Experimental findings determine how subsequent stages of the research evolve.

---

## Current Experimental Evidence

### EXP-001 — Baseline Medical Image Classification

**Status: Complete**

A convolutional neural network was trained and evaluated using the **PneumoniaMNIST** benchmark.

| Metric      | Result |
| ----------- | -----: |
| Accuracy    | 88.46% |
| AUROC       | 0.9370 |
| Sensitivity | 98.46% |
| Specificity | 71.79% |
| Precision   | 85.33% |
| F1-score    | 91.43% |

The model demonstrated strong discrimination and high sensitivity.

However, evaluation also revealed substantially lower specificity and an incorrect prediction associated with approximately **99.98% predicted pneumonia probability**.

This observation motivated a deeper investigation:

> **Can strong predictive performance coexist with unreliable confidence?**

---

### EXP-002 — Probability Calibration

**Status: Complete**

The second experiment investigated whether the baseline model's probability estimates could be interpreted as reliable confidence estimates.

The analysis included:

* reliability analysis;
* Brier score;
* Expected Calibration Error (ECE);
* confidence distributions;
* high-confidence error analysis; and
* temperature scaling.

Temperature scaling was fitted using the validation set to avoid optimising calibration parameters on the held-out test set.

The learned temperature was:

**T = 1.007948**

Validation negative log-likelihood changed from:

**0.097432 → 0.097427**

The fitted temperature remained close to 1.0 and produced negligible improvement in this experimental setting.

This is an important research result rather than a failed experiment.

It motivates the next question:

> **Can explicit uncertainty estimation provide more useful information about unreliable predictions than deterministic probability confidence alone?**

---

### EXP-003 — Uncertainty Quantification

**Status: Planned**

The next phase will investigate whether uncertainty estimates can distinguish between predictions that are relatively reliable and those that are more likely to be incorrect or ambiguous.

The experiment will be designed before implementation, including:

* uncertainty definition;
* estimation method;
* validation procedure;
* uncertainty-error analysis; and
* evaluation criteria.

Results will be added only after the experiment has been completed.

---

## Research Roadmap

| Phase       | Research Investigation                | Status             |
| ----------- | ------------------------------------- | ------------------ |
| EXP-001     | Baseline medical image classification | ✅ Complete         |
| EXP-002     | Probability calibration               | ✅ Complete         |
| EXP-003     | Uncertainty quantification            | ⏳ Planned          |
| EXP-004     | Selective prediction and referral     | ⏳ Planned          |
| EXP-005     | Distribution-shift evaluation         | ⏳ Planned          |
| EXP-006     | Robustness analysis                   | ⏳ Planned          |
| Future      | Multimodal healthcare AI              | Research direction |
| Future      | Trustworthy generative healthcare AI  | Research direction |
| Integration | Trustworthy healthcare AI prototype   | Research direction |

The roadmap is intentionally adaptive. Later experiments may change according to evidence produced by earlier stages.

---

## From Experiments to a Complete System

This repository is not intended to remain a collection of notebooks.

Each research stage investigates a capability that may eventually contribute to an integrated trustworthy healthcare AI research system.

| Research Investigation      | Intended System Capability          |
| --------------------------- | ----------------------------------- |
| Predictive modelling        | Prediction engine                   |
| Probability calibration     | Calibration layer                   |
| Uncertainty quantification  | Uncertainty engine                  |
| Selective prediction        | Risk-aware referral mechanism       |
| Distribution-shift analysis | Shift-awareness evaluation          |
| Robustness analysis         | Reliability testing                 |
| Multimodal modelling        | Multi-source clinical modelling     |
| Generative AI research      | Generative/multimodal AI capability |

Only components supported by experimental evidence will be considered candidates for integration.

---

## Target System Architecture

The longer-term research prototype follows the conceptual pipeline:

```text
            Clinical / Research Inputs
                       │
          ┌────────────┼────────────┐
          │            │            │
        Image      Structured      Text
                     Data
          │            │            │
          └────────────┼────────────┘
                       ▼
                Data Processing
                       │
                       ▼
             Predictive / Multimodal
                     Model
                       │
                       ▼
              Trustworthiness Layer
          ┌────────────┼─────────────┐
          │            │             │
     Calibration   Uncertainty   Robustness /
                                Shift Analysis
          │            │             │
          └────────────┼─────────────┘
                       ▼
                 Risk Assessment
                       │
               ┌───────┴───────┐
               │               │
         Lower-risk         Uncertain /
         prediction        unreliable case
               │               │
               ▼               ▼
          Model output     Flag for review
               │               │
               └───────┬───────┘
                       ▼
                Research Interface
                       │
                       ▼
                 Audit / Monitoring
```

This architecture represents a **research direction**, not a clinically validated system.

---

## Research Principles

The project follows several principles:

**Evidence before claims.** Conclusions are based on completed experiments rather than predetermined outcomes.

**Reproducibility.** Experiments should record their environment, dataset splits, random seeds, configurations, metrics, figures, and relevant model artifacts.

**Separation of validation and testing.** Test data should remain held out from model fitting and calibration optimisation.

**Beyond accuracy.** Evaluation considers discrimination, calibration, uncertainty, robustness, class-specific behaviour, and eventually risk-coverage relationships.

**Negative results matter.** A method that produces negligible improvement can still provide useful evidence and motivate the next research question.

**System components must be earned experimentally.** Planned capabilities are not presented as validated components until supporting experiments have been completed.

---

## Repository Structure

```text
trustworthy-healthcare-ai/
│
├── README.md
│
├── docs/
│   ├── RESEARCH_PLAN.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENTS.md
│   ├── EVALUATION.md
│   ├── REPRODUCIBILITY.md
│   ├── RESEARCH_LOG.md
│   └── PUBLICATION_PLAN.md
│
├── notebooks/
│   ├── 01_baseline_medical_imaging.ipynb
│   └── 02_confidence_calibration.ipynb
│
├── experiments/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── uncertainty/
│
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── tests/
├── pyproject.toml
├── uv.lock
└── .python-version
```

The structure will evolve as new capabilities are implemented.

---

## Reproducibility

The project uses **Python 3.11** and **uv** for environment and dependency management.

Create the environment and install dependencies with:

```bash
uv sync
```

Jupyter can then be started from the managed environment:

```bash
uv run jupyter lab
```

Experiments should be executed using the documented dataset splits, random seeds, and experimental configuration associated with each study.

Detailed reproducibility instructions will be maintained in:

`docs/REPRODUCIBILITY.md`

---

## Documentation

Detailed documentation is organised by purpose:

* **RESEARCH_PLAN.md** — research problem, aims, questions, objectives, scope, and roadmap.
* **SYSTEM_ARCHITECTURE.md** — evolution from experimental components to the integrated research system.
* **METHODOLOGY.md** — datasets, experimental design, modelling procedures, and controls.
* **EXPERIMENTS.md** — registry of planned, active, and completed experiments.
* **EVALUATION.md** — evaluation framework and metric rationale.
* **REPRODUCIBILITY.md** — environment and experiment reproduction instructions.
* **RESEARCH_LOG.md** — chronological research decisions, findings, limitations, and next questions.
* **PUBLICATION_PLAN.md** — pathway from experimental evidence to potential research publication.

---

## Publication Direction

A provisional research direction is:

> **Beyond Accuracy: Evaluating Uncertainty and Robustness in Deep Learning for Medical Image Classification Under Distribution Shift**

This is a working direction rather than a predetermined manuscript title.

The eventual research contribution will be determined by evidence generated from the uncertainty, selective-prediction, robustness, and distribution-shift experiments.

---

## Longer-Term Direction

The initial PneumoniaMNIST experiments provide a controlled environment for studying fundamental trustworthiness problems.

They are not the final objective.

The longer-term research direction is to investigate:

**medical imaging → uncertainty-aware AI → distribution shift → multimodal learning → trustworthy generative healthcare AI**

with the broader question:

> **How can AI systems recognise, quantify, and communicate when their outputs may be unreliable?**

---

## Disclaimer

This repository is intended for **research and educational purposes**.

The models, experiments, and prototype systems developed here are **not medical devices**, have not undergone clinical validation, and must not be used for diagnosis, treatment, or other clinical decision-making.
