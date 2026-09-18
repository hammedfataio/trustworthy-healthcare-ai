# Experiment Registry

## Trustworthy Healthcare AI

**Master Registry for Research Questions, Experimental Evidence, System Decisions, and Research Progression**

---

## 1. Purpose

This document is the master registry for experiments conducted within the Trustworthy Healthcare AI project.

It provides a structured connection between:

**Research Question  
→ Hypothesis  
→ Experiment  
→ Evidence  
→ Interpretation  
→ System Decision  
→ Next Research Question**

The registry serves four purposes:

1. prevent experiments from becoming disconnected notebooks;
2. distinguish completed evidence from planned research;
3. define what evidence is required before an experiment is considered complete; and
4. connect experimental findings to the evolving system architecture.

Future experiments may change as evidence develops.

A planned experiment does not imply a predetermined result.

---

## 2. Experiment Status

Experiments use the following status model:

| Status | Meaning |
|---|---|
| **Complete** | Experiment executed, evaluated, interpreted, documented, and artifacts saved |
| **In Progress** | Experimental implementation or evaluation is underway |
| **Designed** | Research question and methodology defined but implementation has not started |
| **Planned** | Identified as a likely future experiment |
| **Research Direction** | Longer-term investigation whose final experimental design remains open |

---

## 3. Experiment Completion Gate

An experiment is not complete simply because the code executes.

Where applicable, completion requires:

```text
Research Question
        ↓
Hypothesis
        ↓
Experimental Design
        ↓
Implementation
        ↓
Validation
        ↓
Held-Out Evaluation
        ↓
Results
        ↓
Critical Interpretation
        ↓
Limitations
        ↓
Artifacts
        ↓
System Implication
        ↓
Research Log
        ↓
Version-Controlled Commit
```

Only after these requirements are satisfied should an experiment be marked **Complete**.

---

# 4. Master Experiment Roadmap

| ID | Experiment | Primary Question | Status |
|---|---|---|---|
| EXP-001 | Baseline Medical Image Classification | Can a reproducible baseline provide useful discrimination? | **Complete** |
| EXP-002 | Probability Calibration | Can the model's predicted probabilities be interpreted as reliable confidence? | **Complete** |
| EXP-003 | Uncertainty Quantification | Can uncertainty estimates identify potentially unreliable predictions? | **Designed** |
| EXP-004 | Selective Prediction and Referral | Can uncertainty be used to identify cases that should be treated cautiously or referred? | **Planned** |
| EXP-005 | Distribution Shift | Does model reliability deteriorate under changed input conditions, and can the system recognise this? | **Planned** |
| EXP-006 | Robustness | How stable are prediction and reliability signals under controlled perturbations? | **Planned** |
| EXP-007+ | Multimodal Trustworthiness | How does trustworthiness change when multiple clinical modalities are combined? | **Research Direction** |
| Future | Trustworthy Generative AI | How can reliability be evaluated for multimodal and generative healthcare AI outputs? | **Research Direction** |
| Integration | Research System Integration | Can experimentally supported components operate together coherently? | **Research Direction** |

The numbering after EXP-006 is intentionally not fixed.

Results from earlier experiments may justify inserting, modifying, or removing later experiments.

---

# 5. EXP-001 — Baseline Medical Image Classification

## Status

**Complete**

---

## Research Question

> **How effectively can a baseline convolutional neural network distinguish pneumonia-positive from pneumonia-negative chest X-ray images using the PneumoniaMNIST benchmark?**

---

## Purpose

The purpose of EXP-001 was to establish a reproducible predictive baseline before investigating trustworthiness.

Without a baseline, later changes in calibration, uncertainty, robustness, or distribution-shift behaviour would lack a consistent reference point.

---

## Dataset

**PneumoniaMNIST**

Task:

**Normal vs Pneumonia**

Official dataset splits were retained.

| Split | Samples |
|---|---:|
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |

Training class distribution:

| Class | Samples | Share |
|---|---:|---:|
| Normal | 1,214 | 25.79% |
| Pneumonia | 3,494 | 74.21% |

The original imbalance was retained for the baseline experiment.

---

## Model

A lightweight convolutional neural network was used.

Architecture:

```text
Input: 1 × 28 × 28
        ↓
Conv2D: 1 → 16
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2D: 16 → 32
        ↓
ReLU
        ↓
MaxPool
        ↓
Flatten
        ↓
Linear: 1568 → 64
        ↓
ReLU
        ↓
Linear: 64 → 1
        ↓
Logit
```

Training configuration:

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Seed | 42 |

---

## Hypothesis

The baseline CNN was expected to achieve performance above random classification while remaining imperfect enough to expose clinically relevant error patterns.

Strong predictive performance was not assumed to imply reliable confidence.

---

## Held-Out Results

| Metric | Result |
|---|---:|
| Accuracy | 0.884615 |
| AUROC | 0.936993 |
| Sensitivity | 0.9846 |
| Specificity | 0.7179 |
| Precision | 0.8533 |
| F1-score | 0.914286 |

Confusion matrix:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 168 | 66 |
| Actual Pneumonia | 6 | 384 |

---

## Key Observation

The baseline demonstrated strong discrimination and very high sensitivity but substantially lower specificity.

More importantly for the research programme, inspection of individual predictions revealed an incorrect normal case assigned approximately:

**99.98% predicted pneumonia probability**

This observation became the transition point from predictive performance to trustworthiness research.

---

## Interpretation

EXP-001 demonstrated that a model can achieve apparently strong aggregate predictive performance while still producing highly confident individual errors.

Therefore:

> **Accuracy and AUROC alone are insufficient evidence that model confidence is trustworthy.**

This observation motivated EXP-002.

---

## Limitations

The experiment is limited by:

- low-resolution benchmark images;
- a relatively small dataset;
- binary classification;
- class imbalance;
- a lightweight model architecture;
- a single principal training run;
- absence of external validation; and
- absence of clinical workflow evaluation.

---

## System Contribution

EXP-001 establishes the first experimental version of the:

**Prediction Engine**

Status:

**Experimentally Evaluated**

---

## Primary Artifacts

```text
notebooks/01_baseline_medical_imaging.ipynb
docs/experiment_001_baseline.md
results/tables/experiment_001_baseline_metrics.csv
results/models/experiment_001_baseline_cnn.pt
```

Additional figures and supporting artifacts may also be stored under `results/`.

---

## Decision

**Proceed to probability calibration analysis.**

---

# 6. EXP-002 — Probability Calibration

## Status

**Complete**

---

## Research Question

> **How well calibrated are the probability estimates produced by the baseline CNN, and does the model exhibit problematic confidence behaviour when making incorrect predictions?**

---

## Motivation

EXP-001 identified an incorrect prediction associated with extremely high predicted probability.

This raised a new question:

> **When the model reports high confidence, does that confidence correspond meaningfully to observed outcomes?**

---

## Experimental Principle

The predictive model from EXP-001 was retained.

The objective was not to train a better classifier.

The objective was to investigate the reliability of its probability estimates.

---

## Evaluation

EXP-002 included:

- reliability analysis;
- Brier score;
- Expected Calibration Error;
- confidence distributions;
- high-confidence error analysis;
- confidence-accuracy analysis; and
- temperature scaling.

---

## Calibration Protocol

Temperature scaling was fitted using the **validation set**.

The held-out test set was not used to optimise the temperature parameter.

This preserved the separation between:

**calibration fitting → final held-out evaluation**

---

## Temperature Scaling Result

Learned temperature:

**T = 1.007948**

Validation negative log-likelihood:

| Stage | NLL |
|---|---:|
| Before temperature scaling | 0.097432 |
| After temperature scaling | 0.097427 |

The fitted temperature remained close to 1.0.

---

## Test-Set Comparison

Recorded experiment results were:

| Metric | Before | After Temperature Scaling |
|---|---:|---:|
| Brier Score | 0.101087 | 0.101086 |
| ECE | 0.104166 | 0.104166 |
| Accuracy | 0.884615 | 0.884615 |
| AUROC | 0.936993 | 0.936988 |
| F1-score | 0.914286 | 0.914286 |

Changed class predictions:

**0**

---

## Important Interpretation

The result does **not** support the general statement:

> Temperature scaling does not work for medical AI.

The evidence supports the narrower conclusion:

> **For this baseline model and experimental setting, validation-fitted global temperature scaling produced negligible improvement.**

This distinction is important for scientific reporting.

---

## Why Classification Metrics Did Not Change

Positive temperature scaling rescales logits without changing their sign.

For a binary classifier using a probability threshold of 0.5, this generally preserves the predicted class.

Likewise, monotonic scaling preserves prediction ranking, so AUROC should remain effectively unchanged apart from possible numerical effects.

The purpose of temperature scaling is therefore calibration rather than classification improvement.

---

## Methodological Limitations

Calibration results should be interpreted carefully because:

- the test set contains 624 samples;
- ECE depends on binning choices;
- Brier score is not a pure calibration metric;
- the experiment investigates only one post-hoc calibration method;
- the baseline model was trained from a single principal seed; and
- benchmark performance does not establish clinical calibration.

---

## System Contribution

EXP-002 provides evidence for the:

**Calibration and Confidence Evaluation Layer**

However, temperature scaling is **not automatically promoted as the final system calibration method**, because the observed improvement was negligible.

---

## Primary Artifacts

```text
notebooks/02_confidence_calibration.ipynb
docs/experiment_002_calibration.md
results/tables/experiment_002_calibration_comparison.csv
results/figures/experiment_002_reliability_diagram.png
```

---

## Decision

Deterministic probability confidence alone is insufficient for the next stage of the research.

Proceed to:

**EXP-003 — Uncertainty Quantification**

---

# 7. EXP-003 — Uncertainty Quantification

## Status

**Designed — Not Yet Executed**

No experimental results are reported in this section because the experiment has not yet been run.

---

## Research Question

> **Can uncertainty estimates provide useful information about which medical image predictions are more likely to be incorrect or unreliable?**

---

## Motivation

EXP-001 demonstrated high-confidence errors.

EXP-002 demonstrated that simple global temperature scaling produced negligible improvement.

The next research question therefore moves from:

**How confident is the model?**

toward:

**How uncertain should the system be about this particular prediction?**

---

## Core Hypothesis

Predictions associated with greater estimated uncertainty are expected, on average, to contain a higher concentration of errors than predictions associated with lower uncertainty.

This hypothesis must be evaluated rather than assumed.

---

## Important Methodological Distinction

The existing baseline CNN contains no dropout layers.

Therefore, Monte Carlo dropout cannot simply be applied to the existing model and described as uncertainty estimation without changing the architecture.

EXP-003 should distinguish between:

### Deterministic Uncertainty Baseline

Use uncertainty derived from the existing predictive distribution, such as binary predictive entropy.

This establishes whether ordinary deterministic probability already provides useful error-ranking information.

### Explicit Uncertainty Method

A second method may then introduce stochastic or ensemble-based uncertainty, such as:

- a separately trained dropout-enabled model;
- Monte Carlo dropout;
- deep ensembles; or
- another justified method.

This allows a meaningful comparison between deterministic confidence-based uncertainty and a more explicit uncertainty approach.

---

## Candidate Deterministic Measure

For binary probability \(p\), predictive entropy is:

\[
H(p) = -p \log(p) - (1-p)\log(1-p)
\]

Entropy is highest near:

\[
p = 0.5
\]

and lower as probability approaches:

\[
0 \text{ or } 1
\]

However, a highly confident incorrect prediction may still have low entropy.

This is precisely one of the behaviours EXP-003 should investigate.

---

## Proposed Evaluation

EXP-003 should evaluate:

- uncertainty distributions;
- uncertainty for correct predictions;
- uncertainty for incorrect predictions;
- error rate across uncertainty levels;
- whether uncertainty can discriminate correct from incorrect predictions;
- high-confidence / low-uncertainty errors;
- uncertainty ranking behaviour; and
- suitability for downstream selective prediction.

Potential quantitative measures may include:

- AUROC for error detection;
- AUPRC for error detection where appropriate;
- uncertainty summary statistics;
- uncertainty-error plots; and
- ranking-based analysis.

Final metrics should be fixed before final held-out evaluation.

---

## Validation Governance

Any threshold used to define:

- high uncertainty;
- referral;
- abstention; or
- acceptable risk

should be selected using validation data or another predefined procedure.

The held-out test set should not be used to optimise the threshold.

---

## Required Artifacts

Expected artifacts include:

```text
notebooks/03_uncertainty_quantification.ipynb
docs/experiment_003_uncertainty.md
results/tables/experiment_003_uncertainty_metrics.csv
results/figures/experiment_003_uncertainty_distribution.png
```

Additional artifacts should be added only when they are actually produced.

---

## Completion Gate

EXP-003 will not be marked complete until:

- uncertainty method is defined;
- hypothesis is documented;
- implementation is reproducible;
- validation procedure is documented;
- held-out evaluation is completed;
- error-detection behaviour is analysed;
- limitations are documented;
- artifacts are saved; and
- the result is interpreted in the research log.

---

## Intended System Contribution

If experimentally useful, EXP-003 may provide evidence for an:

**Uncertainty Estimation Layer**

Whether the method is integrated will depend on experimental evidence.

---

## Expected Next Decision

If uncertainty successfully ranks prediction risk, proceed to selective prediction.

If it does not, investigate why before introducing a referral mechanism.

---

# 8. EXP-004 — Selective Prediction and Referral

## Status

**Planned**

No results are currently claimed.

---

## Research Question

> **Can uncertainty estimates be used to identify predictions that should be treated cautiously or withheld from ordinary automated output?**

---

## Motivation

Uncertainty is useful only if it informs system behaviour.

EXP-004 therefore converts uncertainty estimation into a decision-oriented research problem.

---

## Concept

```text
Predictions
     ↓
Estimated Risk / Uncertainty
     ↓
Rank Predictions
     ↓
┌─────────────────┬──────────────────┐
│ Lower-risk      │ Higher-risk      │
│ predictions     │ predictions      │
│                 │                  │
│ Retain          │ Refer / Abstain  │
└─────────────────┴──────────────────┘
```

---

## Proposed Evaluation

Potential evaluation includes:

- coverage;
- selective risk;
- risk-coverage curves;
- retained-case accuracy;
- retained-case sensitivity;
- retained-case specificity;
- referral rate; and
- proportion of errors captured among referred cases.

The objective is to determine whether referring uncertain cases improves reliability among retained predictions.

---

## Key Safeguard

Referral thresholds should not be optimised on the held-out test set.

---

## Intended System Contribution

**Risk-Aware Referral / Abstention Layer**

Status:

**Planned**

---

# 9. EXP-005 — Distribution Shift

## Status

**Planned**

---

## Research Question

> **How do predictive performance, calibration, and uncertainty behave when evaluation data differ from the conditions represented during model development?**

---

## Motivation

A trustworthy system should not only perform well under familiar benchmark conditions.

It should also provide useful evidence when its operating conditions change.

---

## Experimental Principle

Distribution shifts should be:

- predefined;
- reproducible;
- parameterised where appropriate; and
- scientifically motivated.

The objective is not simply to create inputs that cause failure.

The experiment should investigate whether:

**performance degradation is accompanied by meaningful changes in uncertainty or reliability signals.**

---

## Proposed Evaluation

Compare in-distribution and shifted conditions using:

- accuracy;
- AUROC;
- sensitivity;
- specificity;
- calibration;
- uncertainty;
- error-detection performance;
- selective risk; and
- coverage where appropriate.

---

## Intended System Contribution

**Shift-Awareness Evaluation Layer**

Status:

**Planned**

---

# 10. EXP-006 — Robustness

## Status

**Planned**

---

## Research Question

> **How stable are predictive performance and trustworthiness signals under controlled input perturbations?**

---

## Potential Perturbations

For medical imaging, later experimental design may consider justified variations in:

- noise;
- contrast;
- brightness;
- blur;
- resolution; or
- other acquisition-related characteristics.

Perturbations should be chosen according to the research question rather than solely according to their ability to damage performance.

---

## Proposed Evaluation

At each perturbation level, evaluate changes in:

- predictive performance;
- calibration;
- uncertainty;
- error detection; and
- selective prediction.

This enables investigation of whether trustworthiness mechanisms degrade alongside the predictive model.

---

## Intended System Contribution

**Robustness Evaluation Layer**

Status:

**Planned**

---

# 11. EXP-007+ — Multimodal Trustworthiness

## Status

**Research Direction**

The final experiment design is intentionally not fixed.

---

## Research Direction

Future experiments may investigate models combining:

- medical imaging;
- structured clinical variables;
- clinical text; and
- other appropriate healthcare modalities.

---

## Potential Research Questions

Future work may investigate:

> Does multimodal fusion improve reliability or only predictive performance?

> What happens when modalities disagree?

> Can uncertainty identify conflicting multimodal evidence?

> What happens when one modality is missing?

> How does distribution shift affecting one modality influence the overall system?

> Is uncertainty dominated by one modality?

---

## Required Baselines

Multimodal experiments should include appropriate unimodal baselines.

This enables comparison between:

**Image only**

**Structured data only**

**Text only**

and

**Multimodal fusion**

where applicable.

---

## Intended System Contribution

**Multimodal Modelling Layer**

Status:

**Research Direction**

---

# 12. Future — Trustworthy Generative Healthcare AI

## Status

**Research Direction**

---

## Motivation

Generative and vision-language systems produce outputs substantially more complex than binary classification probabilities.

Trustworthiness must therefore expand beyond conventional probability calibration.

---

## Potential Research Questions

Future research may investigate:

- uncertainty in generated outputs;
- hallucination or unsupported generation;
- grounding in clinical evidence;
- image-text consistency;
- robustness;
- distribution shift;
- multimodal disagreement;
- confidence communication; and
- identification of outputs requiring review.

---

## Intended System Contribution

**Trustworthy Generative / Vision-Language AI Layer**

Status:

**Research Direction**

---

# 13. System Integration Experiment

## Status

**Research Direction**

Integration should occur only after individual components have sufficient experimental support.

---

## Research Question

> **Can experimentally supported trustworthiness components operate together as a coherent research system without obscuring their individual limitations?**

---

## Target Flow

```text
Healthcare Input
       ↓
Preprocessing
       ↓
Predictive / Multimodal Model
       ↓
Calibration
       ↓
Uncertainty
       ↓
Shift / Robustness Evidence
       ↓
Risk Assessment
       ↓
Prediction OR Review Flag
       ↓
API / Research Interface
       ↓
Monitoring and Audit
```

---

## Integration Principle

A component should not enter the integrated system merely because it was implemented successfully.

Integration requires evidence that the component contributes useful information or behaviour.

---

# 14. Evidence Chain

The research programme currently follows this evidence chain:

```text
EXP-001
Strong predictive performance
+
high-confidence error
        ↓
New question:
Can confidence be trusted?
        ↓
EXP-002
Calibration investigation
+
temperature scaling produced
negligible improvement
        ↓
New question:
Can explicit uncertainty provide
better information about failure?
        ↓
EXP-003
Uncertainty Quantification
        ↓
If informative
        ↓
EXP-004
Selective Prediction
        ↓
Can it survive changing conditions?
        ↓
EXP-005 / EXP-006
Shift + Robustness
        ↓
Extend principles
        ↓
Multimodal / Generative AI
        ↓
Integrated Trustworthy
Healthcare AI Research System
```

This chain should be updated whenever new evidence changes the direction of the research.

---

# 15. Experiment Artifact Standard

Completed experiments should, where applicable, provide:

| Artifact | Purpose |
|---|---|
| Notebook / implementation | Reproduce the experiment |
| Source module | Reuse mature functionality |
| Configuration | Record experimental conditions |
| Model checkpoint | Preserve relevant trained model |
| Metrics table | Preserve quantitative evidence |
| Figure(s) | Communicate important behaviour |
| Experiment report | Explain methodology and findings |
| Research-log entry | Preserve chronological reasoning |
| Git commit | Version the completed research stage |

Not every experiment requires every artifact, but omissions should be intentional.

---

# 16. Experiment Naming Convention

Experiments use:

```text
EXP-001
EXP-002
EXP-003
...
```

Associated files should use corresponding numbering where practical.

Example:

```text
notebooks/03_uncertainty_quantification.ipynb

docs/experiment_003_uncertainty.md

results/tables/experiment_003_uncertainty_metrics.csv

results/figures/experiment_003_uncertainty_distribution.png
```

This creates traceability across code, documentation, and results.

---

# 17. Decision Log Standard

At the end of each experiment, the project should record one of the following:

**Proceed**  
Evidence supports the next planned investigation.

**Refine**  
The method shows promise but requires modification.

**Compare**  
Evidence justifies comparison with alternative approaches.

**Reconsider**  
The hypothesis or method is not sufficiently supported.

**Integrate**  
Evidence supports consideration for the system architecture.

**Do Not Integrate Yet**  
Implementation exists, but evidence is insufficient for system inclusion.

This prevents successful code execution from being confused with successful research evidence.

---

# 18. Current Research Position

At the current stage:

```text
EXP-001  Baseline Classification       COMPLETE
              ↓
EXP-002  Probability Calibration       COMPLETE
              ↓
EXP-003  Uncertainty Quantification    DESIGNED
              ↓
EXP-004  Selective Prediction          PLANNED
              ↓
EXP-005  Distribution Shift            PLANNED
              ↓
EXP-006  Robustness                    PLANNED
              ↓
         Multimodal AI                 RESEARCH DIRECTION
              ↓
         Generative AI                 RESEARCH DIRECTION
              ↓
         System Integration            RESEARCH DIRECTION
```

The immediate research priority is:

# EXP-003 — Uncertainty Quantification

No later experiment should be treated as completed until its evidence exists.

---

# 19. Governing Principle

This registry is intended to prevent the project from becoming a sequence of disconnected technical demonstrations.

Every experiment should answer:

> **What question are we asking?**

> **What evidence would answer it?**

> **What did we actually observe?**

> **What are the limitations?**

> **What should change because of this result?**

The governing principle is:

> **Experiments generate evidence; evidence determines the next research decision; research decisions determine the system architecture.**
