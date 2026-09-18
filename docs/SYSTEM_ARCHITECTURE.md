# System Architecture

## Trustworthy Healthcare AI

**Architecture for an Evidence-Driven, Uncertainty-Aware, Robust, and Multimodal Healthcare AI Research System**

---

## 1. Purpose

This document defines the target architecture for the Trustworthy Healthcare AI project.

The project is not intended to remain a collection of independent machine-learning notebooks. Its longer-term objective is to develop a modular research system in which predictive models are combined with mechanisms for evaluating and communicating model reliability.

The architecture therefore separates:

- predictive modelling;
- probability calibration;
- uncertainty estimation;
- selective prediction and referral;
- distribution-shift evaluation;
- robustness analysis;
- multimodal modelling;
- generative AI;
- system delivery; and
- monitoring and audit.

A central architectural principle is:

> **A model prediction should not automatically be treated as a trustworthy decision simply because the model can produce a confidence score.**

The system is therefore designed around both **prediction** and **reliability assessment**.

---

## 2. Architecture Philosophy

The architecture follows an **evidence-driven integration model**.

Components may appear in the target architecture before they are implemented, but their presence in this document does not imply that they have already been validated.

The development process follows:

**Research Question  
→ Experiment  
→ Evidence  
→ Interpretation  
→ Engineering Decision  
→ System Integration**

This prevents the system architecture from becoming disconnected from the scientific research programme.

A component should only move toward integration when experimental evidence provides a reasonable basis for doing so.

---

## 3. Architecture Status Model

To distinguish implemented work from future plans, components are assigned one of four states.

| Status | Meaning |
|---|---|
| **Implemented** | Code or artifact currently exists in the repository |
| **Experimentally Evaluated** | The component or method has been investigated through a completed experiment |
| **Planned** | Defined in the research roadmap but not yet experimentally completed |
| **Future Research** | Longer-term architectural direction whose final design depends on future evidence |

This distinction is important because the repository represents an evolving research system rather than a finished clinical product.

---

## 4. Current System State

At the current stage, the project has established the initial predictive and calibration research foundation.

### Current evidence

**EXP-001 — Baseline Medical Image Classification**

A CNN baseline has been implemented and evaluated using PneumoniaMNIST.

This provides the first version of the:

**Prediction Engine**

---

**EXP-002 — Probability Calibration**

The baseline model's confidence behaviour has been investigated using reliability analysis, Brier score, Expected Calibration Error, high-confidence error analysis, and temperature scaling.

This provides experimental evidence for the:

**Calibration and Confidence Evaluation Layer**

Temperature scaling produced negligible improvement in the current experimental setting, so it should not automatically be treated as the final calibration mechanism for the integrated system.

---

## 5. Target System Overview

The longer-term architecture is:

```text
                 HEALTHCARE / RESEARCH INPUTS
                            │
              ┌─────────────┼─────────────┐
              │             │             │
            Image       Structured       Text
                          Clinical
                            Data
              │             │             │
              └─────────────┼─────────────┘
                            │
                            ▼
                  DATA PROCESSING LAYER
                            │
                            ▼
                  MODEL / INFERENCE LAYER
                            │
                  ┌─────────┴─────────┐
                  │                   │
             Prediction         Generated /
                                Multimodal
                                  Output
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
                 TRUSTWORTHINESS LAYER
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
       ▼                    ▼                    ▼
   Calibration         Uncertainty         Robustness /
                       Estimation          Shift Analysis
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                            ▼
                    RISK ASSESSMENT
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        Lower-risk output        Uncertain /
                                potentially
                              unreliable output
                │                       │
                ▼                       ▼
         Return model             Flag / Refer
            output                for review
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                  APPLICATION / API LAYER
                            │
                            ▼
                   RESEARCH INTERFACE
                            │
                            ▼
                  MONITORING AND AUDIT
```

This is the **target research architecture**.

It does not imply that every component has already been implemented.

---

## 6. Layer 1 — Healthcare and Research Inputs

The system is designed to evolve from single-modality medical imaging toward multimodal healthcare data.

Potential input modalities include:

### Medical Imaging

Examples may include:

- chest radiographs;
- dental radiographs;
- CT;
- MRI;
- ultrasound; and
- other clinically appropriate imaging modalities.

The current experimental foundation uses PneumoniaMNIST.

### Structured Clinical Data

Future experiments may investigate variables such as:

- demographics;
- measurements;
- laboratory values;
- clinical observations; and
- other structured variables where appropriate datasets are available.

### Clinical Text

Later multimodal or generative experiments may investigate:

- clinical notes;
- reports;
- structured textual descriptions; or
- other appropriately governed textual healthcare data.

The addition of any modality will depend on the research question, dataset suitability, licensing, privacy considerations, and experimental design.

---

## 7. Layer 2 — Data Processing

The data-processing layer converts raw research inputs into representations suitable for modelling.

Potential responsibilities include:

- dataset loading;
- train/validation/test separation;
- image transformations;
- normalisation;
- tensor conversion;
- structured-data preprocessing;
- text preprocessing;
- missing-value handling;
- modality alignment;
- augmentation where experimentally justified; and
- data-quality checks.

The architecture should preserve separation between:

**training data → validation data → held-out test data**

to reduce the risk of evaluation leakage.

Where possible, reusable processing logic should migrate from notebooks into the `src/` codebase as the project matures.

---

## 8. Layer 3 — Model and Inference

The model layer produces the primary predictive or generative output.

### Current Model

The current baseline is a lightweight convolutional neural network for binary pneumonia classification.

Its purpose is not to establish state-of-the-art performance.

It provides a controlled experimental model for investigating trustworthiness properties.

### Future Models

Later research may investigate:

- stronger convolutional architectures;
- pretrained vision models;
- uncertainty-aware architectures;
- multimodal neural networks;
- vision-language models;
- large language models; and
- other generative or multimodal architectures justified by the research question.

Model complexity should increase when scientifically justified rather than simply to make the system larger.

---

## 9. Layer 4 — Trustworthiness Layer

The trustworthiness layer is the central architectural contribution of the project.

Instead of treating the model's raw output as sufficient, the system introduces additional evaluation between model inference and downstream use.

The layer is expected to contain several related components.

---

### 9.1 Probability Calibration

**Purpose:** determine whether predicted probabilities correspond meaningfully to observed outcomes.

Potential methods may include:

- temperature scaling;
- alternative post-hoc calibration methods; and
- calibration-aware modelling approaches.

Current evidence from EXP-002 shows that global temperature scaling produced negligible improvement for the existing baseline.

Therefore, the architecture does not assume temperature scaling is the final calibration solution.

**Current status: Experimentally Evaluated**

---

### 9.2 Uncertainty Quantification

**Purpose:** estimate how uncertain the system is about individual predictions.

EXP-003 will investigate whether uncertainty estimates provide useful information beyond deterministic confidence.

Potential approaches may include:

- predictive entropy;
- stochastic inference;
- Monte Carlo dropout where architecturally appropriate;
- ensemble-based uncertainty;
- or other uncertainty methods supported by the literature and experimental design.

The specific method should be selected before implementation and justified scientifically.

**Current status: Planned**

---

### 9.3 Selective Prediction and Referral

A trustworthy system should not necessarily treat every prediction equally.

Selective prediction investigates whether the system can identify predictions for which automated output should be treated cautiously.

Conceptually:

```text
Prediction
    │
    ▼
Uncertainty / Risk Estimate
    │
    ├──────── Low estimated risk ────────► Return prediction
    │
    └──────── High estimated risk ───────► Flag for review
```

This creates a potential **abstention or referral mechanism**.

The research question is not simply whether uncertainty can be calculated, but whether it is sufficiently informative to improve system behaviour when uncertain cases are handled differently.

Potential evaluation may include:

- coverage;
- selective risk;
- risk-coverage curves;
- retained-case performance; and
- error concentration among referred cases.

Referral thresholds should be determined using validation data or another appropriately designed procedure rather than optimised on the held-out test set.

**Current status: Planned**

---

### 9.4 Distribution-Shift Evaluation

Healthcare data encountered after model development may differ from the data used during training.

Distribution-shift experiments will investigate:

> **What happens when the model encounters data that differ systematically from its original training conditions?**

The evaluation should examine not only predictive degradation but also whether:

- calibration changes;
- uncertainty increases;
- failure detection improves or deteriorates; and
- selective prediction remains useful.

This connects robustness research directly to the trustworthiness layer.

**Current status: Planned**

---

### 9.5 Robustness Analysis

Robustness experiments will evaluate model behaviour when inputs are altered under controlled conditions.

Depending on the research question, this may include appropriate variations in:

- image quality;
- noise;
- contrast;
- acquisition-like perturbations;
- missing information; or
- other relevant transformations.

The objective is not simply to create difficult examples.

The objective is to understand:

> **How does model reliability change as input conditions move away from the conditions under which the model was developed?**

**Current status: Planned**

---

## 10. Layer 5 — Risk Assessment

The risk-assessment layer combines evidence from the predictive and trustworthiness components.

A future risk assessment may consider:

```text
Prediction
    +
Predicted Probability
    +
Calibration Information
    +
Uncertainty
    +
Shift / Robustness Signals
    ↓
Risk-Aware System Behaviour
```

The exact combination should not be predetermined before supporting experiments are completed.

The research programme will determine which signals are genuinely informative.

This layer is therefore currently an architectural target rather than an implemented decision mechanism.

**Current status: Future Research**

---

## 11. Layer 6 — Referral / Abstention

A central future capability is the ability to distinguish between:

**predictions suitable for ordinary model output**

and

**predictions that warrant additional review because the system lacks sufficient evidence for reliability.**

Conceptually:

```text
                    Model Output
                         │
                         ▼
                  Reliability Check
                         │
              ┌──────────┴──────────┐
              │                     │
        Acceptable Risk         Elevated Risk
              │                     │
              ▼                     ▼
       Return Prediction       Flag for Review
```

This mechanism should not be described as a clinical referral system unless and until it is evaluated in an appropriate clinical context.

Within this project, it represents a **research prototype for selective prediction and risk-aware AI behaviour**.

---

## 12. Layer 7 — Multimodal AI

The architecture is designed to evolve beyond a single image input.

A future multimodal system may combine:

```text
Medical Image ─────┐
                   │
Clinical Data ─────┼────► Multimodal Representation
                   │
Clinical Text ─────┘
                            │
                            ▼
                    Predictive /
                    Generative Model
```

Multimodal research introduces additional trustworthiness questions.

For example:

- What happens when modalities disagree?
- How should uncertainty be represented across modalities?
- What happens when one modality is missing?
- Does one modality dominate the prediction?
- Does multimodal fusion improve reliability or merely predictive performance?
- How does distribution shift affect individual modalities?
- Can uncertainty help detect conflicting evidence?

These questions provide a bridge between the initial medical imaging experiments and the longer-term trustworthy generative AI direction.

**Current status: Future Research**

---

## 13. Layer 8 — Generative and Vision-Language AI

Generative models introduce challenges that differ from binary classification.

A classifier may produce:

```text
Pneumonia probability = 0.82
```

A generative model may produce:

```text
Image + clinical context
        ↓
Generated interpretation
        ↓
Clinical-language response
```

Trustworthiness must therefore eventually extend beyond probability calibration.

Future research may investigate:

- uncertainty in generated outputs;
- hallucination or unsupported generation;
- multimodal consistency;
- robustness under distribution shift;
- confidence communication;
- factual grounding;
- evaluation frameworks for generated clinical content; and
- mechanisms for identifying outputs requiring review.

This stage represents the longer-term transition toward **trustworthy generative healthcare AI**.

**Current status: Future Research**

---

## 14. Layer 9 — Application and API Layer

Once research components become sufficiently stable, reusable system interfaces can be introduced.

A future architecture may include:

```text
Research Interface
        │
        ▼
       API
        │
        ▼
Inference Service
        │
        ▼
Prediction Model
        │
        ▼
Trustworthiness Engine
        │
        ▼
Risk-Aware Response
```

Potential engineering responsibilities include:

- request validation;
- inference orchestration;
- model loading;
- structured responses;
- error handling;
- logging;
- version tracking; and
- service health checks.

This layer should be built after the scientific behaviour of the core components is sufficiently understood.

**Current status: Future Engineering**

---

## 15. Layer 10 — Research Interface

A future research interface may expose:

- input data;
- model prediction;
- predicted probability;
- uncertainty estimate;
- calibration information;
- risk/referral status;
- model/version information; and
- relevant experimental metadata.

The interface should prioritise transparent presentation of model behaviour rather than presenting the system as clinically validated software.

A possible conceptual output is:

```text
Prediction: Pneumonia

Predicted probability: 0.84
Uncertainty: Elevated
Reliability status: Review recommended

Model version: ...
Experiment configuration: ...
```

The exact presentation will be determined later and should reflect the evidence produced by the research.

**Current status: Future Engineering**

---

## 16. Layer 11 — Monitoring and Audit

Trustworthiness does not end when a prediction is generated.

A mature research system should eventually support auditability.

Potential monitoring signals may include:

- model version;
- input characteristics;
- prediction distributions;
- confidence distributions;
- uncertainty distributions;
- referral frequency;
- potential shift indicators;
- latency;
- system errors; and
- experimental configuration.

The purpose is to make model behaviour traceable and support future analysis.

Privacy-sensitive healthcare information should not be logged indiscriminately.

Monitoring design must consider data minimisation, privacy, security, and governance requirements.

**Current status: Future Engineering**

---

## 17. Research-to-System Mapping

Each experiment contributes evidence toward one or more architectural components.

| Experiment | Research Question | Architectural Contribution | Status |
|---|---|---|---|
| EXP-001 | Can the baseline model make useful predictions? | Prediction engine | Complete |
| EXP-002 | Are its probabilities reliable? | Calibration evaluation | Complete |
| EXP-003 | Can uncertainty identify unreliable predictions? | Uncertainty engine | Planned |
| EXP-004 | Can uncertain cases be handled selectively? | Risk/referral layer | Planned |
| EXP-005 | What happens when the data distribution changes? | Shift-awareness evaluation | Planned |
| EXP-006 | How stable is the model under controlled perturbation? | Robustness layer | Planned |
| Future | Can multiple clinical modalities be combined reliably? | Multimodal engine | Research direction |
| Future | Can generative outputs be evaluated for trustworthiness? | Generative AI layer | Research direction |
| Integration | Can validated components operate together? | Integrated research prototype | Research direction |

This table provides the connection between the scientific programme and the engineering roadmap.

---

## 18. Proposed Code Architecture

As reusable components emerge from the experiments, the source tree may evolve toward:

```text
src/
├── data/
│
├── models/
│
├── calibration/
│
├── uncertainty/
│
├── robustness/
│
├── shift/
│
├── evaluation/
│
├── inference/
│
├── api/
│
├── monitoring/
│
└── utils/
```

This represents a target structure.

Directories should be introduced when the corresponding functionality exists rather than creating empty architecture solely for appearance.

---

## 19. Experiment Artifact Flow

Every major experiment should ideally produce traceable artifacts.

```text
Research Question
       │
       ▼
Experimental Configuration
       │
       ▼
Implementation / Notebook
       │
       ▼
Model Artifact
       │
       ▼
Metrics + Figures
       │
       ▼
Interpretation
       │
       ▼
Experiment Documentation
       │
       ▼
Architecture Decision
```

Where appropriate, artifacts may include:

- notebook;
- reusable source code;
- configuration;
- model checkpoint;
- CSV metrics;
- figures;
- experiment report; and
- research-log entry.

This provides traceability from scientific question to system decision.

---

## 20. Reproducibility Architecture

Reproducibility is treated as part of the system rather than an afterthought.

The project currently uses:

- Python 3.11;
- `uv` dependency management;
- version-controlled source code;
- documented random seeds;
- explicit dataset splits;
- saved result tables;
- saved model artifacts where appropriate; and
- Git-based research history.

As the project matures, reproducibility should include explicit experiment configuration and increasingly reusable execution paths outside notebooks.

---

## 21. Research and Engineering Boundaries

The project deliberately separates three concepts.

### Research Prototype

Used to test scientific hypotheses and investigate model behaviour.

### Engineering Prototype

Used to demonstrate how experimentally supported components can operate together as a system.

### Clinical System

A system intended for actual clinical use would require substantially more evidence and governance, potentially including external validation, clinical evaluation, security controls, regulatory consideration, human-factors evaluation, data governance, and deployment-specific monitoring.

This repository currently targets the first two categories.

It does **not** claim to provide a clinically deployable medical system.

---

## 22. Privacy and Security Direction

As the project progresses toward richer healthcare data and system integration, privacy and security become increasingly important.

Future architecture should consider:

- data minimisation;
- de-identification where applicable;
- access control;
- secure secret management;
- input validation;
- dependency security;
- audit logging;
- model and dataset provenance;
- privacy-preserving learning where scientifically relevant; and
- protection against inappropriate disclosure of healthcare information.

Security features should be implemented when required by the system stage rather than merely listed as completed capabilities.

---

## 23. Testing Strategy

As functionality moves from notebooks into reusable modules, the project should progressively introduce:

### Unit Tests

For individual functions and components.

### Integration Tests

For interactions between:

- preprocessing;
- model inference;
- calibration;
- uncertainty;
- risk assessment; and
- API components.

### Research Validation Tests

Where practical, tests may verify:

- expected tensor dimensions;
- deterministic behaviour under controlled seeds;
- valid probability ranges;
- correct dataset separation;
- checkpoint loading; and
- metric calculations.

### System Tests

Later stages may test complete inference flows from input through trustworthiness assessment to structured output.

---

## 24. Observability Direction

For the eventual research system, observability should distinguish between software health and model behaviour.

### Software Observability

Potential signals include:

- request failures;
- inference latency;
- service availability; and
- exceptions.

### Model Observability

Potential signals include:

- prediction distributions;
- confidence distributions;
- uncertainty distributions;
- referral rates;
- potential data drift;
- model version; and
- performance where ground truth becomes available.

This distinction will become increasingly important as the project transitions from experiments to an integrated prototype.

---

## 25. Architecture Evolution

The architecture is expected to evolve through approximately four stages.

### Stage 1 — Experimental Foundation

**Current stage**

Focus:

- baseline medical imaging;
- calibration;
- reproducibility;
- experimental documentation.

### Stage 2 — Trustworthiness Engine

Focus:

- uncertainty quantification;
- selective prediction;
- distribution shift;
- robustness.

Expected outcome:

A reusable trustworthiness layer supported by experimental evidence.

### Stage 3 — Multimodal Research System

Focus:

- image + structured clinical information;
- potentially clinical text;
- multimodal uncertainty;
- cross-modal reliability.

Expected outcome:

A research system capable of studying trustworthiness across multiple modalities.

### Stage 4 — Trustworthy Generative AI Prototype

Focus:

- vision-language or generative models;
- uncertainty-aware output;
- grounding;
- robustness;
- risk-aware response;
- API/interface integration;
- monitoring and audit.

Expected outcome:

An integrated research prototype for investigating trustworthy multimodal and generative healthcare AI.

---

## 26. Target End-State

The intended end-state is not simply:

```text
Input
  ↓
Model
  ↓
Prediction
```

The project aims toward:

```text
Healthcare Input
       ↓
Data Processing
       ↓
Predictive / Generative Model
       ↓
Trustworthiness Assessment
       ↓
Calibration + Uncertainty + Robustness + Shift Evidence
       ↓
Risk-Aware Decision
       ↓
Output OR Flag for Review
       ↓
Research Interface / API
       ↓
Monitoring + Audit
```

This architecture reflects the central principle of the project:

> **A trustworthy AI system should provide evidence not only about what it predicts, but also about when that prediction may be unreliable.**

---

## 27. Current Architecture Status

At the time of this architecture definition:

| Component | Status |
|---|---|
| Medical image baseline | Experimentally evaluated |
| Prediction pipeline | Implemented experimentally |
| Calibration evaluation | Experimentally evaluated |
| Temperature scaling | Experimentally evaluated; negligible improvement in current setting |
| Uncertainty quantification | Planned |
| Selective prediction | Planned |
| Distribution-shift evaluation | Planned |
| Robustness evaluation | Planned |
| Risk-assessment engine | Future research |
| Multimodal modelling | Future research |
| Generative / vision-language modelling | Future research |
| API layer | Future engineering |
| Research interface | Future engineering |
| Monitoring / audit | Future engineering |

This table should be updated as the research programme progresses.

---

## 28. Architecture Principle

The architecture is intentionally ambitious, but implementation remains evidence-driven.

The system will not become trustworthy merely because calibration, uncertainty, robustness, or multimodal components are added to a diagram.

Each component must be investigated experimentally.

The architecture therefore follows one governing principle:

> **Research evidence determines system design — not the other way around.**
