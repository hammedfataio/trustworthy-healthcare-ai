# Research Plan

## Trustworthy Healthcare AI

**Research and Engineering Framework for Uncertainty-Aware, Robust, and Multimodal AI for Healthcare Decision Support**

---

## 1. Research Overview

Artificial intelligence systems can achieve strong predictive performance on medical datasets while still producing unreliable confidence estimates, failing under changes in data distribution, or making highly confident incorrect predictions.

In healthcare, these limitations are particularly important because predictive accuracy alone does not establish whether an AI system can be relied upon in unfamiliar, ambiguous, or high-risk situations.

This project investigates the development and systematic evaluation of trustworthy healthcare AI, with particular emphasis on:

- medical image analysis;
- probability calibration;
- uncertainty quantification;
- selective prediction and referral;
- robustness;
- distribution shift;
- multimodal learning; and
- trustworthy generative AI.

The project combines a structured scientific research programme with the engineering of a reproducible AI research system.

Rather than treating individual machine-learning experiments as isolated notebooks, each experiment addresses a specific research question and contributes evidence toward the design of the wider system.

---

## 2. Research Motivation

A model can make many correct predictions and still behave unreliably.

For example, a healthcare AI model may:

- achieve high accuracy while performing poorly for a clinically important class;
- assign very high confidence to an incorrect prediction;
- produce probabilities that do not correspond well to observed outcomes;
- become unreliable when deployment data differ from training data;
- fail to communicate when an input is unfamiliar or ambiguous; or
- continue producing apparently confident outputs even when its performance is degrading.

This creates an important distinction between **predictive performance** and **predictive reliability**.

The research therefore moves beyond the question:

> **Can the model make accurate predictions?**

toward the broader question:

> **Can the model provide useful evidence about when its predictions should and should not be trusted?**

---

## 3. Central Research Problem

Many machine-learning systems are optimised primarily for predictive performance.

However, high-stakes applications such as healthcare require additional properties.

A trustworthy system should ideally provide evidence about:

1. how well it predicts;
2. whether its probability estimates are calibrated;
3. how uncertain it is about individual predictions;
4. whether uncertainty is associated with prediction errors;
5. how it behaves when the input distribution changes;
6. how robust its predictions are to perturbations;
7. whether potentially unreliable cases can be identified for additional review; and
8. whether these properties remain meaningful as models become multimodal or generative.

This project investigates these properties incrementally through controlled experiments.

---

## 4. Central Research Question

> **How can uncertainty-aware and robust machine-learning methods improve the reliability of healthcare AI systems, particularly when models encounter difficult cases or changes in data distribution?**

As the project develops toward multimodal and generative AI, the broader research question becomes:

> **How can healthcare AI systems integrate multiple clinical data modalities while providing meaningful uncertainty and reliability information that supports safer decision-making?**

---

## 5. Research Aim

The primary aim is to develop and systematically evaluate a reproducible framework for trustworthy healthcare AI that combines predictive modelling with:

- probability calibration;
- uncertainty estimation;
- selective prediction;
- robustness analysis;
- distribution-shift evaluation; and
- risk-aware decision support.

The longer-term aim is to extend these principles from controlled medical image classification experiments toward multimodal and generative healthcare AI.

---

## 6. Research Objectives

The project will pursue the following objectives:

1. Establish reproducible baseline models for medical prediction tasks.

2. Evaluate predictive performance using multiple appropriate metrics rather than accuracy alone.

3. Investigate whether predicted probabilities correspond meaningfully to observed outcomes.

4. Evaluate approaches for estimating predictive uncertainty.

5. Determine whether uncertainty estimates can help identify incorrect, ambiguous, or potentially unreliable predictions.

6. Investigate selective prediction and referral strategies in which uncertain cases can be flagged for additional review.

7. Evaluate model behaviour under controlled forms of distribution shift.

8. Investigate robustness under relevant input perturbations.

9. Determine whether uncertainty remains informative when model performance deteriorates.

10. Extend the research toward models capable of combining multiple clinical data modalities.

11. Investigate how calibration, uncertainty, robustness, and risk-aware evaluation apply to multimodal and generative healthcare AI.

12. Integrate experimentally supported components into a reproducible trustworthy healthcare AI research prototype.

---

## 7. Research Philosophy

The project follows an **evidence-driven experimental approach**.

Future experiments may be planned in advance, but their conclusions are not predetermined.

Each research stage follows the cycle:

**Research Question  
→ Hypothesis  
→ Experimental Design  
→ Implementation  
→ Evaluation  
→ Interpretation  
→ Limitations  
→ System Implication  
→ Next Research Question**

Experimental findings may therefore modify later stages of the roadmap.

A method that produces negligible or negative improvement is not automatically considered a failed experiment.

If the experiment is appropriately designed, such a result may provide useful evidence about the limitations of that method and motivate the next research question.

---

## 8. Current Experimental Foundation

### EXP-001 — Baseline Medical Image Classification

**Status: Complete**

The first experiment established a reproducible convolutional neural network baseline using the PneumoniaMNIST benchmark.

The model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.8846 |
| AUROC | 0.9370 |
| Sensitivity | 0.9846 |
| Specificity | 0.7179 |
| Precision | 0.8533 |
| F1-score | 0.9143 |

The experiment demonstrated strong discrimination and high sensitivity while also revealing lower specificity.

Inspection of individual predictions identified an incorrect example assigned approximately **99.98% predicted pneumonia probability**.

This observation demonstrated why predictive performance alone is insufficient for evaluating model reliability and motivated the next research question:

> **Can the probability estimates produced by the model actually be trusted as confidence estimates?**

---

### EXP-002 — Probability Calibration

**Status: Complete**

The second experiment investigated the reliability of the baseline model's probability estimates.

The evaluation included:

- reliability analysis;
- Brier score;
- Expected Calibration Error (ECE);
- confidence distributions;
- high-confidence error analysis; and
- temperature scaling.

Temperature scaling was fitted using the validation set rather than the held-out test set.

The learned temperature was:

**T = 1.007948**

Validation negative log-likelihood changed from:

**0.097432 → 0.097427**

The fitted temperature remained very close to 1.0 and produced negligible improvement in the validation objective.

This result should not be interpreted as evidence that temperature scaling is generally ineffective.

Rather, it indicates that **global temperature scaling produced negligible improvement for this model and experimental setting**.

The result motivates a deeper question:

> **Can explicit uncertainty estimation provide more useful information about unreliable predictions than deterministic probability confidence alone?**

---

## 9. Research Roadmap

The current experimental programme is:

| Phase | Investigation | Status |
|---|---|---|
| EXP-001 | Baseline medical image classification | Complete |
| EXP-002 | Probability calibration | Complete |
| EXP-003 | Uncertainty quantification | Planned |
| EXP-004 | Selective prediction and referral | Planned |
| EXP-005 | Distribution-shift evaluation | Planned |
| EXP-006 | Robustness analysis | Planned |
| Future | Multimodal healthcare AI | Research direction |
| Future | Trustworthy generative healthcare AI | Research direction |
| Integration | Trustworthy healthcare AI research prototype | Research direction |

This roadmap is intentionally adaptive.

Experiments may be modified, added, reordered, or removed when evidence from earlier stages justifies doing so.

---

## 10. From Research Questions to System Capabilities

The project is not intended to remain a collection of notebooks.

Each research stage investigates a capability that may eventually contribute to an integrated research system.

| Research Investigation | Intended System Capability |
|---|---|
| Baseline modelling | Prediction engine |
| Probability calibration | Calibration layer |
| Uncertainty quantification | Uncertainty estimation |
| Selective prediction | Risk-aware referral |
| Distribution-shift analysis | Shift-awareness evaluation |
| Robustness analysis | Reliability testing |
| Multimodal modelling | Multi-source clinical modelling |
| Generative AI research | Trustworthy generative/multimodal capability |

A planned capability is not considered validated simply because it appears in the architecture.

Only methods supported by experimental evidence will become candidates for integration into the research prototype.

---

## 11. Target System Concept

The longer-term research system follows the conceptual pipeline:

**Clinical / Research Inputs  
→ Data Processing  
→ Predictive or Multimodal Model  
→ Trustworthiness Layer  
→ Risk Assessment  
→ Model Output or Review Recommendation  
→ Monitoring and Audit**

The trustworthiness layer is expected to investigate:

- probability calibration;
- predictive uncertainty;
- distribution-shift behaviour;
- robustness; and
- risk-aware prediction.

The intended system behaviour is therefore not simply:

**Input → Prediction**

but:

**Input → Prediction → Reliability Assessment → Risk-Aware Output**

A sufficiently uncertain prediction may eventually be treated differently from a lower-risk prediction.

This provides the conceptual basis for selective prediction and referral.

---

## 12. Research and System Tracks

The project contains three connected tracks.

### Research Track

Investigates scientific questions through controlled experiments.

This includes:

- calibration;
- uncertainty;
- selective prediction;
- distribution shift;
- robustness; and
- multimodal trustworthiness.

### Engineering Track

Transforms experimentally supported methods into reusable system components.

This may eventually include:

- reusable model services;
- inference pipelines;
- trustworthiness components;
- APIs;
- experiment tracking;
- testing;
- monitoring; and
- a research interface.

### Publication Track

Transforms sufficiently rigorous experimental findings into research communication.

This includes:

- research questions;
- experimental evidence;
- quantitative analysis;
- figures and tables;
- limitations;
- related literature;
- manuscript development; and
- potential preprint or publication submission.

The three tracks are intended to converge into a coherent research portfolio rather than operate independently.

---

## 13. Evaluation Framework

The project will not rely on a single metric.

Depending on the experiment, evaluation may include:

### Predictive Performance

- accuracy;
- AUROC;
- sensitivity;
- specificity;
- precision; and
- F1-score.

### Probability Quality and Calibration

- negative log-likelihood;
- Brier score;
- Expected Calibration Error;
- reliability diagrams; and
- confidence analysis.

### Uncertainty

Future experiments may evaluate:

- predictive entropy;
- uncertainty distributions;
- uncertainty-error relationships;
- discrimination between correct and incorrect predictions; and
- uncertainty under shifted inputs.

### Selective Prediction

Evaluation may include:

- coverage;
- selective risk;
- risk-coverage relationships; and
- performance after referring uncertain predictions.

### Robustness and Distribution Shift

Evaluation may examine:

- predictive degradation;
- calibration degradation;
- uncertainty behaviour;
- failure detection; and
- performance across controlled shifts.

Specific evaluation procedures will be defined before interpreting each experiment wherever practical.

---

## 14. Reproducibility Principles

Experiments should be reproducible from the repository.

The project therefore aims to maintain:

- dependency management using `uv`;
- controlled Python versions;
- documented random seeds;
- clearly defined dataset splits;
- separation between training, validation, and test data;
- version-controlled source code;
- saved experimental metrics;
- saved figures;
- model checkpoints where appropriate;
- documented experimental configurations; and
- written interpretations of results.

The held-out test set should not be used for model fitting, calibration parameter optimisation, or experimental tuning.

---

## 15. Definition of a Completed Experiment

An experiment is not considered complete simply because a notebook executes successfully.

Where applicable, completion requires:

**Research Question  
→ Hypothesis  
→ Experimental Design  
→ Implementation  
→ Results  
→ Quantitative Evaluation  
→ Critical Interpretation  
→ Limitations  
→ Reproducibility Check  
→ System Implication  
→ Documentation**

This definition is intended to maintain consistency across the research programme.

---

## 16. Research Outputs

The project is designed to produce several complementary outputs.

### Research Evidence

Reproducible experimental evidence concerning calibration, uncertainty, selective prediction, robustness, distribution shift, and later multimodal AI.

### Research Software

A structured codebase implementing reusable experimental and system components.

### Research Prototype

An integrated prototype demonstrating how prediction and trustworthiness components can operate together.

### Research Communication

Technical documentation, experiment reports, figures, tables, and research summaries.

### Publication Development

A body of evidence that can be assessed for its suitability for a research manuscript or preprint.

### PhD Research Portfolio

A transparent record of research formulation, implementation, evaluation, critical analysis, and scientific progression.

---

## 17. Current Publication Direction

A provisional research direction is:

> **Beyond Accuracy: Evaluating Uncertainty and Robustness in Deep Learning for Medical Image Classification Under Distribution Shift**

This is a working direction rather than a predetermined manuscript title.

The final research contribution will depend on evidence generated by the uncertainty, selective-prediction, robustness, and distribution-shift experiments.

The research question may therefore evolve if the experimental evidence reveals a more meaningful contribution.

---

## 18. Scope and Limitations

The initial experiments use a relatively small benchmark medical imaging dataset and lightweight neural architectures.

These experiments provide a controlled environment for investigating trustworthiness concepts.

They should not be interpreted as evidence of clinical deployment readiness.

The project does not currently claim:

- clinical validation;
- regulatory approval;
- diagnostic superiority to clinicians;
- generalisation across healthcare populations;
- production clinical deployment readiness; or
- state-of-the-art predictive performance.

Later stages will progressively investigate more challenging modelling and evaluation conditions.

---

## 19. Longer-Term Multimodal and Generative Direction

The initial medical imaging experiments establish the experimental foundation.

The longer-term direction is:

**Medical Imaging  
→ Calibration  
→ Uncertainty  
→ Selective Prediction  
→ Distribution Shift  
→ Robustness  
→ Multimodal Healthcare AI  
→ Trustworthy Generative AI**

Future multimodal research may investigate combinations of:

- medical images;
- structured clinical variables;
- clinical text; and
- other appropriate healthcare modalities.

Generative and vision-language models introduce additional trustworthiness questions because their outputs may be more complex than a single classification probability.

The longer-term research question therefore becomes:

> **How can multimodal and generative healthcare AI systems recognise, quantify, and communicate when their outputs may be unreliable?**

---

## 20. Project Position

This repository represents an evolving **research and engineering programme**, not a finished clinical product.

Its purpose is to develop experimental evidence, reusable research infrastructure, and a progressively integrated prototype for investigating trustworthy healthcare AI.

The guiding principle throughout the project is:

> **High predictive performance is useful, but trustworthy healthcare AI also requires evidence about when a model may be wrong.**
