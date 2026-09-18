# Publication Plan

## Trustworthy Healthcare AI

**Evidence-to-Publication Strategy for Uncertainty-Aware and Robust Medical AI**

---

## 1. Purpose

This document defines the publication strategy for the Trustworthy Healthcare AI research programme.

The objective is to transform experimentally supported findings from the repository into a coherent research manuscript rather than treating publication as a separate activity performed after experimentation.

The publication strategy follows:

```text
Research Question
        ↓
Experiment
        ↓
Evidence
        ↓
Critical Analysis
        ↓
Research Contribution
        ↓
Manuscript
        ↓
Preprint / Submission
```

The central principle is:

> **The manuscript will be shaped by the evidence produced by the experiments rather than by a predetermined conclusion.**

---

## 2. Current Publication Direction

The current provisional research direction is:

> **Beyond Accuracy: Evaluating Uncertainty and Robustness in Deep Learning for Medical Image Classification Under Distribution Shift**

This is a working direction rather than a fixed final title.

The final title should reflect the strongest supported contribution after the uncertainty, selective-prediction, distribution-shift, and robustness experiments are completed.

---

## 3. Research Motivation

Medical AI models are frequently evaluated using predictive performance measures such as accuracy and AUROC.

These metrics are important but do not fully characterise whether a model behaves reliably.

A model may:

- achieve strong discrimination;
- produce highly confident errors;
- exhibit imperfect calibration;
- fail to recognise unfamiliar inputs;
- remain overconfident under distribution shift; or
- provide uncertainty estimates that do not correspond meaningfully to failure.

The research therefore investigates trustworthiness beyond predictive accuracy.

---

## 4. Emerging Research Question

The current publication programme is developing around the broader question:

> **To what extent do calibration and uncertainty measures provide useful evidence about model reliability, particularly when medical image classification models encounter errors or changing input conditions?**

This question may evolve as experimental evidence develops.

---

## 5. Potential Contribution

A publication should not claim novelty merely because several standard methods were implemented.

A meaningful contribution must emerge from the experimental evidence.

Potential contribution areas include:

### Reliability Beyond Accuracy

Demonstrating why strong predictive discrimination does not necessarily imply trustworthy confidence.

### High-Confidence Failure Analysis

Characterising situations where incorrect predictions remain associated with extreme model confidence.

### Calibration Limitations

Evaluating whether standard post-hoc calibration meaningfully improves reliability in the chosen experimental setting.

### Uncertainty as Failure Detection

Testing whether uncertainty provides useful information for identifying model errors.

### Selective Prediction

Investigating whether uncertainty-based abstention can reduce risk among retained predictions.

### Distribution Shift

Evaluating whether uncertainty responds appropriately when model performance deteriorates under changed conditions.

### Robustness

Investigating how prediction, calibration, and uncertainty behave across controlled perturbations.

The final contribution should be selected only after sufficient evidence exists.

---

## 6. Evidence Already Available

### EXP-001 — Baseline Medical Image Classification

Current held-out results:

| Metric | Result |
|---|---:|
| Accuracy | 0.8846 |
| AUROC | 0.9370 |
| Sensitivity | 0.9846 |
| Specificity | 0.7179 |
| Precision | 0.8533 |
| F1-score | 0.9143 |

A particularly important observation was an incorrect prediction assigned approximately:

**99.98% probability of pneumonia**

This provides motivation for studying confidence reliability.

---

### EXP-002 — Probability Calibration

Temperature scaling fitted using validation data produced:

**Learned temperature: 1.007948**

Validation NLL:

```text
Before: 0.097432
After:  0.097427
```

The improvement was negligible.

The current evidence therefore supports the limited conclusion:

> **Validation-fitted global temperature scaling produced negligible improvement for the baseline model under the current experimental conditions.**

This result should not be generalised beyond the experimental setting.

---

## 7. Evidence Still Required

A credible paper requires substantially more evidence than EXP-001 and EXP-002 alone currently provide.

The highest-priority missing evidence is:

```text
EXP-003
Uncertainty Quantification
        ↓
EXP-004
Selective Prediction
        ↓
EXP-005
Distribution Shift
        ↓
EXP-006
Robustness
```

The manuscript should not be finalised before the central uncertainty and robustness claims have experimental support.

---

# 8. EXP-003 Publication Role

EXP-003 investigates:

> **Can uncertainty identify unreliable predictions?**

Potential publication evidence includes:

- uncertainty distributions for correct and incorrect predictions;
- error-detection AUROC;
- error-detection AUPRC;
- high-confidence / low-uncertainty errors;
- uncertainty ranking behaviour; and
- comparison of uncertainty approaches where justified.

This experiment is likely to become a central component of the manuscript.

---

# 9. EXP-004 Publication Role

EXP-004 investigates whether uncertainty can improve system behaviour through selective prediction.

Potential evidence includes:

- risk-coverage curves;
- retained-case performance;
- referral rate;
- proportion of errors captured by referral; and
- performance at predefined coverage levels.

This moves the paper from:

**uncertainty measurement**

toward:

**uncertainty-informed decision behaviour**.

---

# 10. EXP-005 Publication Role

EXP-005 introduces distribution shift.

This may become one of the strongest parts of the publication because trustworthy AI is particularly important when deployment conditions differ from development data.

Potential evidence includes:

```text
In-Distribution
       vs
Shifted
```

for:

- discrimination;
- calibration;
- uncertainty;
- error detection; and
- selective prediction.

A particularly important question is:

> **Does uncertainty increase when predictive reliability deteriorates?**

---

# 11. EXP-006 Publication Role

EXP-006 evaluates controlled robustness.

Potential evidence includes performance trajectories across perturbation severity.

For example:

```text
Perturbation Severity
        ↓
Prediction Performance
        ↓
Calibration
        ↓
Uncertainty
        ↓
Error Detection
```

This can reveal whether trustworthiness mechanisms fail before, alongside, or after predictive performance.

---

# 12. Candidate Manuscript Narrative

If supported by the final evidence, the manuscript may follow the narrative:

```text
Strong classification performance
        ↓
does not guarantee
        ↓
Reliable confidence
        ↓
Calibration alone may be insufficient
        ↓
Explicit uncertainty is evaluated
        ↓
Uncertainty is tested as an error signal
        ↓
Uncertainty informs selective prediction
        ↓
System is challenged by distribution shift
        ↓
Reliability behaviour is evaluated
        ↓
Implications for trustworthy medical AI
```

This narrative must remain evidence-dependent.

---

# 13. Candidate Paper Structure

A future manuscript may follow:

## Abstract

Research problem, methodology, key findings, contribution, and implications.

## 1. Introduction

Motivation for trustworthy medical AI and limitations of accuracy-focused evaluation.

## 2. Related Work

Research on:

- medical image classification;
- calibration;
- uncertainty quantification;
- selective prediction;
- distribution shift;
- robustness; and
- trustworthy AI.

## 3. Methodology

Dataset, model, experimental design, calibration, uncertainty methods, shift construction, robustness procedures, and evaluation metrics.

## 4. Baseline Predictive Performance

EXP-001.

## 5. Probability Calibration

EXP-002.

## 6. Uncertainty Quantification

EXP-003.

## 7. Selective Prediction

EXP-004.

## 8. Distribution Shift

EXP-005.

## 9. Robustness Analysis

EXP-006.

## 10. Discussion

Interpretation across experiments.

## 11. Limitations

Explicit methodological and generalisation limitations.

## 12. Conclusion

Evidence-supported conclusions and future research.

The final manuscript structure may change according to results.

---

# 14. Figure Plan

Potential publication figures include:

### Figure 1 — Research Framework

```text
Prediction
    ↓
Calibration
    ↓
Uncertainty
    ↓
Selective Prediction
    ↓
Distribution Shift
    ↓
Robustness
```

### Figure 2 — Baseline Performance

Potential confusion matrix and/or ROC representation.

### Figure 3 — Calibration

Reliability diagram before and after calibration.

### Figure 4 — Uncertainty and Error

Distribution of uncertainty for correct and incorrect predictions.

### Figure 5 — Error Detection

ROC or precision-recall analysis using uncertainty as the error-detection signal.

### Figure 6 — Risk-Coverage

Selective risk versus coverage.

### Figure 7 — Distribution Shift

Performance, calibration, and uncertainty across shift severity.

### Figure 8 — Robustness

Reliability behaviour across controlled perturbations.

Only figures that contribute directly to the research argument should appear in the final manuscript.

---

# 15. Table Plan

Potential publication tables include:

### Table 1 — Dataset Characteristics

Dataset size, split, class distribution, modality, resolution.

### Table 2 — Baseline Model

Architecture and training configuration.

### Table 3 — Baseline Results

Predictive performance metrics.

### Table 4 — Calibration Results

Before/after calibration comparison.

### Table 5 — Uncertainty Results

Error-detection and uncertainty metrics.

### Table 6 — Selective Prediction

Performance across selected coverage levels.

### Table 7 — Distribution Shift

Comparison across shift conditions.

### Table 8 — Robustness

Performance across perturbation levels.

Tables should be generated from saved experiment artifacts wherever practical.

---

# 16. Literature Strategy

The literature review should develop alongside the experiments.

Core literature themes include:

```text
Trustworthy AI
      │
      ├── Medical AI
      ├── Calibration
      ├── Uncertainty Quantification
      ├── Selective Prediction
      ├── Distribution Shift
      ├── Robustness
      ├── Multimodal AI
      └── Generative Healthcare AI
```

Priority should be given to:

- peer-reviewed research;
- highly relevant methodological papers;
- major benchmark papers;
- systematic reviews where useful;
- recent work from strong research groups; and
- papers directly related to the methods being evaluated.

---

# 17. Literature Tracking

For each important paper, useful information to record includes:

```text
Citation
Research Problem
Dataset
Method
Evaluation
Main Finding
Limitation
Connection to This Project
```

This prevents the literature review from becoming a list of unrelated summaries.

---

# 18. Research Gap Development

The final research gap should not be artificially declared before the literature and experiments are sufficiently developed.

Instead:

```text
Literature
    +
Experimental Evidence
    +
Observed Limitations
        ↓
Defensible Research Gap
```

Potential gaps may involve the relationship between:

- predictive performance;
- calibration;
- uncertainty;
- failure detection;
- selective prediction; and
- behaviour under distribution shift.

The precise gap should be refined as the evidence develops.

---

# 19. Avoiding Overclaiming

The manuscript should avoid statements such as:

```text
"Our model is clinically reliable."

"Our method guarantees trustworthy AI."

"This system is ready for clinical use."

"Temperature scaling does not work in healthcare."

"Our uncertainty method detects all unsafe predictions."
```

unless extraordinary evidence supports such claims.

Preferred language should reflect the actual scope of the experiments.

For example:

> **Under the evaluated benchmark conditions...**

> **The results suggest...**

> **In this experimental setting...**

> **The method was associated with...**

> **Further external validation is required...**

---

# 20. Negative Results

Negative or negligible results should remain part of the research record.

EXP-002 demonstrates this principle.

Temperature scaling produced negligible improvement, but this result remains useful because it motivated a deeper investigation into uncertainty and model failure.

A scientifically meaningful negative result should not be removed simply because it does not improve performance.

---

# 21. Reproducibility and Publication

Publication results should be traceable to repository artifacts.

Conceptually:

```text
Paper Claim
    ↓
Table / Figure
    ↓
Saved Result Artifact
    ↓
Evaluation Code
    ↓
Experiment
    ↓
Configuration
    ↓
Dataset + Model
```

This connection is essential for credible computational research.

---

# 22. Repository-to-Paper Mapping

| Repository Component | Publication Role |
|---|---|
| `RESEARCH_PLAN.md` | Research direction |
| `METHODOLOGY.md` | Experimental methodology |
| `EXPERIMENTS.md` | Experiment registry |
| `EVALUATION.md` | Metric justification |
| `REPRODUCIBILITY.md` | Reproduction framework |
| Experiment reports | Detailed experimental evidence |
| Notebooks | Experimental implementation |
| `results/tables/` | Quantitative evidence |
| `results/figures/` | Visual evidence |
| Research log | Evolution of reasoning |

The publication should therefore emerge from the repository rather than being disconnected from it.

---

# 23. Publication Evidence Gate

Before drafting a full manuscript, the project should ideally have evidence addressing:

```text
[✓] Baseline predictive performance

[✓] Calibration behaviour

[ ] Explicit uncertainty evaluation

[ ] Error-detection performance

[ ] Selective prediction

[ ] Distribution shift

[ ] Robustness

[ ] Repeated-run / variability analysis where required

[ ] Final statistical analysis

[ ] Final figures

[ ] Final limitations analysis
```

The exact gate may change as the contribution becomes clearer.

---

# 24. Writing Strategy

The manuscript should be developed incrementally.

### Current Stage

Maintain:

- research motivation;
- experiment records;
- methodology;
- literature notes; and
- candidate contribution.

### After EXP-003

Draft:

- uncertainty methodology;
- uncertainty results;
- initial central findings.

### After EXP-004

Add:

- selective prediction;
- risk-coverage analysis.

### After EXP-005 / EXP-006

Develop:

- shift analysis;
- robustness results;
- integrated discussion.

### Final Stage

Complete:

- abstract;
- final introduction;
- final related work;
- discussion;
- limitations;
- conclusion; and
- supplementary material where required.

The abstract should be finalised late because it must reflect the actual results.

---

# 25. Preprint Strategy

Once the research contains a defensible contribution and the manuscript reaches suitable quality, a preprint may provide:

- a public research output;
- a citable research artifact;
- evidence of independent research activity;
- material for supervisor discussions; and
- a foundation for later peer-reviewed submission.

A preprint should not be released solely to satisfy a portfolio deadline if the experimental evidence is incomplete.

---

# 26. Venue Selection

A publication venue should be selected after the contribution and scope become clear.

Potential venue categories may include:

- medical AI;
- biomedical informatics;
- machine learning for healthcare;
- trustworthy AI;
- medical imaging;
- AI safety/reliability workshops; and
- relevant interdisciplinary venues.

Venue choice should depend on:

- contribution;
- methodological depth;
- dataset scope;
- experimental strength;
- paper maturity; and
- venue requirements.

A prestigious venue should not be selected solely for its name if the manuscript does not match its scope.

---

# 27. Supervisor Alignment

The publication direction intentionally develops capabilities relevant to research in:

- trustworthy AI;
- uncertainty quantification;
- multimodal AI;
- medical imaging;
- distribution shift;
- robustness; and
- generative healthcare AI.

The objective is not to imitate a specific research group.

The objective is to demonstrate the ability to:

> identify a research problem, design experiments, produce evidence, interpret limitations, build reproducible research software, and communicate findings.

These capabilities are directly relevant to doctoral research.

---

# 28. Relationship to the Complete System

The publication and engineering tracks remain connected but distinct.

```text
Research Experiments
       │
       ├──────────────► Publication Evidence
       │
       └──────────────► Validated System Components
                              ↓
                    Integrated Research Prototype
```

Not every engineering feature belongs in the paper.

Not every research experiment must become a deployed component.

The connection is evidence.

---

# 29. Longer-Term Publication Direction

The initial publication focuses primarily on the trustworthy behaviour of predictive medical AI.

Later work may extend toward:

```text
Medical Imaging
      ↓
Uncertainty
      ↓
Distribution Shift
      ↓
Multimodal AI
      ↓
Vision-Language Models
      ↓
Trustworthy Generative Healthcare AI
```

This creates the possibility of multiple future research contributions rather than attempting to place every research question into one manuscript.

---

# 30. Current Publication Position

Current evidence:

```text
EXP-001   COMPLETE
Baseline prediction
        ↓
EXP-002   COMPLETE
Calibration
        ↓
EXP-003   NEXT
Uncertainty
        ↓
EXP-004
Selective prediction
        ↓
EXP-005
Distribution shift
        ↓
EXP-006
Robustness
        ↓
Publication-ready evidence package
```

The immediate publication priority is therefore not writing the final paper.

It is:

> **Produce rigorous EXP-003 evidence.**

---

# 31. Success Criteria

The publication track will be considered successful when the project can present:

- a clearly defined research problem;
- a defensible research gap;
- reproducible methodology;
- meaningful experimental evidence;
- appropriate baselines;
- uncertainty-aware evaluation;
- robustness or shift analysis;
- transparent negative results;
- clearly stated limitations;
- reproducible figures and tables;
- a coherent contribution; and
- a manuscript whose conclusions do not exceed its evidence.

---

# 32. Governing Publication Principle

The purpose of publication is not to make the project appear successful.

The purpose is to communicate what the experiments reveal.

Therefore:

> **The evidence determines the claim.**

Not:

> **The desired claim determines which evidence is reported.**

This principle governs the publication strategy for the Trustworthy Healthcare AI project.
