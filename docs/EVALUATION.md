# Evaluation Framework

## Trustworthy Healthcare AI

**Evaluation Framework for Predictive Performance, Calibration, Uncertainty, Selective Prediction, Distribution Shift, Robustness, and Multimodal Generative AI**

---

## 1. Purpose

This document defines the evaluation framework for the Trustworthy Healthcare AI research programme.

The project deliberately evaluates more than predictive accuracy.

A model may achieve strong classification performance while remaining unreliable because it may:

- produce highly confident incorrect predictions;
- provide poorly calibrated probabilities;
- fail to express increased uncertainty when incorrect;
- deteriorate under distribution shift;
- remain overconfident on unfamiliar inputs; or
- produce unsupported outputs in future generative settings.

The evaluation framework therefore separates several dimensions of trustworthy behaviour.

The central evaluation principle is:

> **No single metric is sufficient to establish that an AI system is trustworthy.**

---

## 2. Evaluation Dimensions

The research evaluates model behaviour across the following dimensions:

| Dimension | Primary Question |
|---|---|
| Predictive performance | Does the model make useful predictions? |
| Discrimination | Can the model separate positive and negative cases? |
| Calibration | Do predicted probabilities correspond to observed outcomes? |
| Uncertainty | Does uncertainty provide information about possible model failure? |
| Error detection | Can reliability signals distinguish incorrect from correct predictions? |
| Selective prediction | Does withholding higher-risk predictions improve retained performance? |
| Distribution shift | What happens when evaluation conditions differ from development conditions? |
| Robustness | How stable is model behaviour under controlled perturbation? |
| Multimodal reliability | How does reliability change when multiple modalities are combined? |
| Generative reliability | Are generated outputs grounded, consistent, and appropriately uncertain? |

These dimensions are evaluated separately before conclusions are combined.

---

# 3. Evaluation Philosophy

The project follows four evaluation principles.

### 3.1 Evaluate the Research Question

Metrics should be selected because they answer the research question rather than because they produce favourable numbers.

### 3.2 Use Multiple Complementary Metrics

Metrics often capture different aspects of model behaviour.

For example:

**AUROC ≠ calibration**

and:

**calibration ≠ uncertainty**

and:

**uncertainty ≠ robustness**

Strong performance in one dimension should not be presented as evidence of another.

### 3.3 Preserve Held-Out Evaluation

Method selection, calibration fitting, uncertainty thresholds, and referral thresholds should use validation data or another predefined development procedure.

Final evaluation should use held-out data.

### 3.4 Report Negative Results

A method that produces negligible improvement remains scientifically informative when the experimental design is valid.

EXP-002 is an example: global temperature scaling produced negligible improvement under the current experimental conditions.

---

# 4. Predictive Performance Evaluation

Predictive evaluation establishes whether the underlying model provides a meaningful predictive baseline.

For binary classification, the project currently considers:

- accuracy;
- AUROC;
- sensitivity;
- specificity;
- precision;
- F1-score; and
- confusion matrix.

---

## 4.1 Accuracy

Accuracy measures:

\[
\text{Accuracy} =
\frac{TP + TN}
{TP + TN + FP + FN}
\]

It answers:

> What proportion of predictions were correct?

### Strength

Simple and interpretable.

### Limitation

Accuracy can be misleading under class imbalance.

In the current PneumoniaMNIST training split, pneumonia cases are substantially more frequent than normal cases.

Therefore, accuracy should not be interpreted alone.

---

## 4.2 Sensitivity

Sensitivity measures:

\[
\text{Sensitivity} =
\frac{TP}{TP + FN}
\]

It measures the proportion of actual positive cases correctly identified.

For the current pneumonia task:

> Of the pneumonia-positive cases, how many were detected?

EXP-001 produced approximately:

**Sensitivity = 0.9846**

This indicates that relatively few positive cases were missed in that held-out evaluation.

It does not establish calibration or overall trustworthiness.

---

## 4.3 Specificity

Specificity measures:

\[
\text{Specificity} =
\frac{TN}{TN + FP}
\]

It measures the proportion of negative cases correctly identified.

EXP-001 produced approximately:

**Specificity = 0.7179**

This was substantially lower than sensitivity.

The difference demonstrates why class-specific metrics are necessary.

---

## 4.4 Precision

Precision measures:

\[
\text{Precision} =
\frac{TP}{TP + FP}
\]

It answers:

> Among predictions classified as positive, what proportion were actually positive?

EXP-001 produced approximately:

**Precision = 0.8533**

Precision depends partly on class prevalence and should therefore be interpreted in the context of the evaluated population.

---

## 4.5 F1-Score

F1 combines precision and recall:

\[
F1 =
2 \times
\frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
\]

EXP-001 produced:

**F1 = 0.9143**

F1 is useful under class imbalance but does not incorporate true negatives directly and does not measure probability quality.

---

# 5. Discrimination Evaluation

## AUROC

The Area Under the Receiver Operating Characteristic Curve evaluates how effectively the model ranks positive examples above negative examples across decision thresholds.

EXP-001 produced:

**AUROC = 0.9370**

This indicates strong discrimination in the held-out benchmark evaluation.

However:

> **High AUROC does not imply calibrated probabilities.**

A model may rank examples correctly while assigning inappropriate probability values.

This distinction motivated the transition from EXP-001 to EXP-002.

---

# 6. Confusion Matrix Analysis

For binary classification:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | TN | FP |
| Actual Positive | FN | TP |

EXP-001 produced:

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 168 | 66 |
| Actual Pneumonia | 6 | 384 |

The confusion matrix provides information that aggregate scores can conceal.

It showed that the baseline generated substantially more false positives than false negatives.

---

# 7. Probability Calibration Evaluation

Calibration asks:

> **When a model predicts a probability, how well does that probability correspond to observed outcomes?**

A model that assigns approximately 0.8 probability to many comparable cases would ideally be correct at a frequency consistent with that probability, subject to sampling variability and the definition of the predicted event.

Calibration evaluation currently includes:

- reliability diagrams;
- Expected Calibration Error;
- Brier score;
- confidence analysis; and
- high-confidence error analysis.

---

# 8. Reliability Diagrams

Reliability diagrams compare predicted probability with observed outcome frequency across probability regions.

Conceptually:

```text
Predicted Probability
        ↓
Group Predictions
        ↓
Observed Outcome Frequency
        ↓
Compare Prediction vs Observation
```

A well-calibrated model should approximately follow the ideal calibration relationship.

However, reliability diagrams depend on:

- sample size;
- binning strategy;
- number of bins; and
- distribution of predictions across bins.

They should therefore be interpreted alongside quantitative measures.

---

# 9. Expected Calibration Error

Expected Calibration Error summarises differences between confidence and observed accuracy across bins.

A common form is:

\[
ECE =
\sum_{m=1}^{M}
\frac{|B_m|}{n}
\left|
\text{acc}(B_m) -
\text{conf}(B_m)
\right|
\]

where:

- \(B_m\) represents a confidence bin;
- \(M\) is the number of bins; and
- \(n\) is the total number of predictions.

Lower values generally indicate smaller observed calibration gaps under the chosen binning procedure.

### Important Limitation

ECE is sensitive to:

- bin count;
- bin boundaries;
- sample size; and
- implementation details.

Therefore:

> **ECE should not be interpreted as an absolute measure of trustworthiness.**

The exact binning strategy should be documented.

---

# 10. Brier Score

For binary outcomes:

\[
\text{Brier} =
\frac{1}{N}
\sum_{i=1}^{N}
(p_i-y_i)^2
\]

where:

- \(p_i\) is predicted probability; and
- \(y_i\) is the observed binary outcome.

Lower Brier scores indicate lower squared probabilistic prediction error.

However:

> **The Brier score is a proper scoring rule, not a pure measure of calibration.**

It reflects aspects of probabilistic prediction quality beyond calibration alone.

It should therefore be interpreted together with reliability analysis and other metrics.

---

# 11. High-Confidence Error Analysis

Aggregate calibration can conceal important individual failures.

The project therefore explicitly investigates:

> **Incorrect predictions made with high confidence.**

EXP-001 revealed an incorrect normal case assigned approximately:

**99.98% probability of pneumonia**

Such examples are important because a model may appear well-performing at population level while remaining dangerously overconfident on particular errors.

High-confidence error analysis will therefore remain part of later experiments.

---

# 12. Temperature Scaling Evaluation

EXP-002 evaluated validation-fitted temperature scaling.

Learned temperature:

**T = 1.007948**

Validation NLL:

```text
Before: 0.097432
After:  0.097427
```

The change was negligible.

This provides evidence about the current model and experimental setting only.

It does not establish that temperature scaling is generally ineffective.

---

# 13. Uncertainty Evaluation

EXP-003 extends evaluation from confidence toward uncertainty.

The primary question becomes:

> **Does greater estimated uncertainty correspond to greater likelihood of model error?**

A useful uncertainty measure should ideally provide information about model failure.

Simply producing an uncertainty number is not sufficient.

---

# 14. Correct vs Incorrect Uncertainty

One initial analysis will compare uncertainty distributions between:

```text
Correct Predictions
        vs
Incorrect Predictions
```

If uncertainty is informative, incorrect predictions may tend to exhibit higher uncertainty.

However, distributional differences alone are insufficient.

The ability to rank or identify errors should also be evaluated quantitatively.

---

# 15. Error Detection as a Secondary Task

Prediction correctness can be reframed as a secondary evaluation problem.

For each prediction:

```text
Correct   → non-error
Incorrect → error
```

The uncertainty score can then be evaluated as an **error-detection signal**.

This allows direct investigation of:

> Can uncertainty distinguish model failures from successful predictions?

---

# 16. AUROC for Error Detection

AUROC can be used to evaluate whether uncertainty ranks errors above correct predictions.

The interpretation differs from disease-classification AUROC.

Here:

```text
Higher uncertainty
        ↓
Should ideally correspond to
        ↓
Greater probability of model error
```

A useful error-detection AUROC would indicate that uncertainty contains information about prediction correctness.

This metric should always be clearly labelled:

**Error-Detection AUROC**

to avoid confusion with disease-classification AUROC.

---

# 17. AUPRC for Error Detection

Incorrect predictions may be less frequent than correct predictions.

Therefore, precision-recall analysis may provide additional information.

AUPRC can evaluate how effectively uncertainty identifies the less frequent error class.

Its interpretation depends strongly on error prevalence.

The baseline error rate should therefore be reported alongside AUPRC.

---

# 18. Predictive Entropy

For binary prediction probability \(p\):

\[
H(p) =
-p\log(p)
-
(1-p)\log(1-p)
\]

Entropy is greatest near:

\[
p = 0.5
\]

and approaches zero as:

\[
p \rightarrow 0
\]

or:

\[
p \rightarrow 1
\]

This creates an important limitation.

A highly confident incorrect prediction may have:

**low predictive entropy**

despite being wrong.

EXP-003 should explicitly investigate this failure mode.

---

# 19. Future Stochastic Uncertainty Evaluation

If a stochastic uncertainty method is introduced, additional quantities may become available.

Examples include:

- predictive variance;
- predictive entropy across stochastic predictions;
- mutual information;
- ensemble disagreement; and
- variation across stochastic forward passes.

The exact measures should depend on the uncertainty method selected.

They should not be added simply because they are available.

---

# 20. Selective Prediction Evaluation

Selective prediction asks:

> **Can system reliability improve if higher-risk predictions are withheld or referred?**

This introduces the concept of:

**coverage**

and

**selective risk**.

---

# 21. Coverage

Coverage represents the proportion of predictions retained by the system.

Conceptually:

\[
\text{Coverage}
=
\frac{\text{Retained Predictions}}
{\text{All Predictions}}
\]

Example:

If 90 of 100 predictions are retained:

\[
\text{Coverage} = 0.90
\]

Lower coverage means more predictions are referred or withheld.

---

# 22. Selective Risk

Selective risk measures prediction error among retained cases.

For classification, one simple formulation is:

\[
\text{Selective Risk}
=
1 -
\text{Accuracy on Retained Cases}
\]

A useful selective-prediction system should ideally demonstrate:

```text
Lower Coverage
      ↓
Higher-risk cases removed
      ↓
Lower Error Among Retained Cases
```

This relationship must be measured rather than assumed.

---

# 23. Risk-Coverage Curve

The risk-coverage curve evaluates selective performance across multiple uncertainty thresholds.

Conceptually:

```text
Coverage
100% ─────────────────────────────
 90% ───────────────────────
 80% ─────────────────
 70% ───────────
             ↓
     Measure retained risk
```

The curve provides more information than selecting one arbitrary referral threshold.

---

# 24. Referral Effectiveness

EXP-004 should investigate whether referred cases contain a disproportionate share of model errors.

Potential measures include:

- proportion of errors captured by referral;
- referral rate;
- retained-case accuracy;
- retained sensitivity;
- retained specificity; and
- risk-coverage behaviour.

A useful referral mechanism should not be judged solely by improved retained accuracy.

The cost of reduced coverage must also be reported.

---

# 25. Distribution-Shift Evaluation

Distribution-shift evaluation compares:

```text
In-Distribution Behaviour
           vs
Shifted Behaviour
```

The project should evaluate whether shift affects:

- predictive performance;
- calibration;
- uncertainty;
- error detection;
- selective prediction; and
- confidence behaviour.

---

# 26. Relative Performance Degradation

For metric \(M\), change under shift can be reported explicitly.

For example:

\[
\Delta M =
M_{\text{shifted}} -
M_{\text{reference}}
\]

The direction of desirable change depends on the metric.

Reporting both absolute values and differences helps communicate degradation clearly.

---

# 27. Uncertainty Response to Shift

A particularly important research question is:

> **When model performance deteriorates, does estimated uncertainty increase?**

A model that performs worse under shift while remaining equally confident may present a more serious trustworthiness problem than a model whose uncertainty appropriately increases.

Therefore, shift evaluation should compare:

```text
Performance Change
        +
Calibration Change
        +
Uncertainty Change
        +
Error-Detection Change
```

---

# 28. Robustness Evaluation

Robustness experiments should evaluate behaviour across controlled perturbation severity.

Example conceptual design:

```text
Original Input
      ↓
Perturbation Level 1
      ↓
Perturbation Level 2
      ↓
Perturbation Level 3
      ↓
Compare:
Prediction
Calibration
Uncertainty
Error Detection
```

The evaluation should avoid reporting only the most extreme perturbation.

The trajectory of degradation is itself informative.

---

# 29. Robustness Curves

Where appropriate, results may be plotted as:

```text
Metric
  │
  │\
  │ \
  │  \
  │   \
  └──────────────
    Perturbation
      Severity
```

Different metrics may degrade at different rates.

For example, discrimination may remain relatively stable while calibration deteriorates.

This distinction is important for trustworthy-AI evaluation.

---

# 30. Repeated Runs

A single training run provides limited information about experimental variability.

Where computationally feasible and scientifically important, later experiments should include multiple seeds.

Results may then be reported as:

\[
\text{Mean} \pm \text{Standard Deviation}
\]

or with appropriate confidence intervals.

Repeated experiments become especially important when comparing methods whose differences are small.

---

# 31. Confidence Intervals

Point estimates should not always be treated as exact characteristics of model performance.

Where appropriate, confidence intervals may be estimated using methods such as:

- bootstrap resampling; or
- analytically justified procedures.

The chosen procedure should match the metric and experimental design.

Confidence intervals can help distinguish potentially meaningful differences from sampling variability.

---

# 32. Statistical Comparison

Statistical tests should only be introduced when they answer a clearly defined research question.

Possible future comparisons may involve:

- baseline vs alternative uncertainty method;
- in-distribution vs shifted performance;
- calibrated vs uncalibrated probability quality;
- unimodal vs multimodal models; or
- uncertainty distributions for correct vs incorrect predictions.

Statistical significance alone should not be interpreted as practical or clinical significance.

---

# 33. Multimodal Evaluation

Future multimodal experiments should evaluate both predictive improvement and reliability.

Potential comparison:

| Model | Prediction | Calibration | Uncertainty | Robustness |
|---|---|---|---|---|
| Image only | Evaluate | Evaluate | Evaluate | Evaluate |
| Structured data only | Evaluate | Evaluate | Evaluate | Evaluate |
| Text only | Evaluate where applicable | Evaluate | Evaluate | Evaluate |
| Multimodal | Evaluate | Evaluate | Evaluate | Evaluate |

The central question is not simply:

> Does multimodal AI improve accuracy?

It is also:

> **Does multimodal AI improve or degrade reliability?**

---

# 34. Multimodal Ablation

Ablation experiments can investigate the contribution of individual modalities.

Conceptually:

```text
Full Multimodal Model
        ↓
Remove Image
        ↓
Remove Structured Data
        ↓
Remove Text
        ↓
Compare Behaviour
```

This can reveal whether:

- one modality dominates;
- modalities provide complementary information;
- reliability depends heavily on one source; or
- conflicting modalities create uncertainty.

---

# 35. Missing-Modality Evaluation

Healthcare data may be incomplete.

Future multimodal experiments should therefore consider:

> What happens when an expected modality is unavailable?

Evaluation may compare:

```text
All Modalities Available
          vs
One Modality Missing
          vs
Multiple Modalities Missing
```

Relevant outcomes include:

- predictive degradation;
- uncertainty response;
- calibration change; and
- system referral behaviour.

---

# 36. Generative AI Evaluation

Generative healthcare AI requires evaluation beyond conventional classification metrics.

Future evaluation may consider:

- factual correctness;
- grounding;
- unsupported claims;
- consistency;
- multimodal alignment;
- robustness;
- uncertainty communication;
- response completeness; and
- appropriateness of review/referral behaviour.

The final evaluation framework will depend on the specific generative task.

---

# 37. Grounding Evaluation

For future vision-language or generative systems, outputs should be evaluated against the evidence supplied to the model.

The question becomes:

> **Is the generated statement supported by the available input?**

This is distinct from linguistic fluency.

A response may sound convincing while being unsupported.

Therefore:

> **Fluency should never be used as a proxy for factual reliability.**

---

# 38. Hallucination / Unsupported Generation

Future generative experiments should distinguish between:

- supported statements;
- incorrect statements;
- unsupported statements;
- omitted relevant information; and
- appropriately uncertain responses.

Evaluation procedures should be defined before final comparison.

---

# 39. Human Evaluation

Some future generative or multimodal tasks may require human evaluation.

If human evaluation is introduced, the study design should specify:

- evaluator expertise;
- evaluation criteria;
- rating procedure;
- blinding where appropriate;
- disagreement handling;
- inter-rater reliability where relevant; and
- ethical or governance requirements.

Human evaluation should not be introduced casually as a substitute for measurable evaluation.

---

# 40. Evaluation Under Class Imbalance

Because the current dataset is imbalanced, the project should avoid relying on accuracy alone.

Evaluation should include metrics that expose class-specific behaviour.

At minimum, current binary experiments should consider:

```text
Accuracy
+
AUROC
+
Sensitivity
+
Specificity
+
Precision
+
F1
+
Confusion Matrix
```

Later experiments may add other metrics where justified.

---

# 41. Error Analysis

Quantitative evaluation should be complemented by structured error analysis.

Potential categories include:

- false positives;
- false negatives;
- high-confidence errors;
- low-uncertainty errors;
- high-uncertainty correct predictions;
- shifted-input failures; and
- repeated qualitative failure patterns.

Error analysis can generate new hypotheses.

However, observations discovered after examining the test set should be treated as exploratory unless independently validated.

---

# 42. Avoiding Metric Cherry-Picking

A method should not be declared successful simply because one metric improves.

Suppose a method produces:

```text
Accuracy       ↑
Calibration    ↓
Uncertainty    unchanged
Robustness     ↓
```

The correct interpretation is not automatically:

**The method is better.**

Instead, the result indicates a trade-off requiring further analysis.

The project should report important favourable and unfavourable outcomes.

---

# 43. Evaluation Decision Matrix

Future system integration decisions may use the following reasoning framework:

| Evidence | Possible Decision |
|---|---|
| Clear improvement across relevant measures | Consider integration |
| Small improvement within likely variability | Repeat / investigate |
| Improvement in one dimension but degradation in another | Analyse trade-off |
| No meaningful improvement | Do not integrate solely for complexity |
| Worse performance | Investigate or reject |
| Unclear evidence | Collect additional evidence |

This is a reasoning framework, not an automatic scoring system.

---

# 44. System-Level Evaluation

When components are eventually integrated, evaluation should move beyond isolated modules.

The complete research prototype should be assessed across:

```text
Input
  ↓
Prediction
  ↓
Calibration
  ↓
Uncertainty
  ↓
Risk Assessment
  ↓
Referral Behaviour
  ↓
Output
```

System-level questions include:

- Are predictions accurate?
- Are probabilities meaningful?
- Does uncertainty identify failures?
- Does referral reduce retained risk?
- Does the system recognise distribution shift?
- Does performance degrade gracefully?
- Are outputs traceable?
- Are limitations communicated?

---

# 45. Current Evaluation Baseline

At the current research stage:

### Predictive Evaluation

**Completed through EXP-001**

Results include:

- Accuracy: 0.8846
- AUROC: 0.9370
- Sensitivity: 0.9846
- Specificity: 0.7179
- Precision: 0.8533
- F1: 0.9143

### Calibration Evaluation

**Completed through EXP-002**

Global temperature scaling produced negligible improvement in the current experimental setting.

### Uncertainty Evaluation

**Next research stage — EXP-003**

### Selective Prediction

**Planned — EXP-004**

### Distribution Shift

**Planned — EXP-005**

### Robustness

**Planned — EXP-006**

### Multimodal / Generative Evaluation

**Future research direction**

---

# 46. Evaluation Progression

The evaluation programme therefore progresses as:

```text
Can the model predict?
        ↓
EXP-001
        ↓
Can its probabilities be trusted?
        ↓
EXP-002
        ↓
Can uncertainty identify failure?
        ↓
EXP-003
        ↓
Can uncertainty improve system behaviour?
        ↓
EXP-004
        ↓
Does reliability survive distribution change?
        ↓
EXP-005
        ↓
Does reliability survive perturbation?
        ↓
EXP-006
        ↓
Can these principles extend to multimodal AI?
        ↓
Future Experiments
        ↓
Can they extend to generative healthcare AI?
```

---

# 47. Reporting Standard

Every experiment should report:

### Primary Research Question

What is being tested?

### Primary Metrics

Which measures directly answer the question?

### Secondary Metrics

Which measures provide additional context?

### Baseline

What is the comparison point?

### Validation Procedure

How were decisions made without contaminating the test set?

### Held-Out Results

What happened under final evaluation?

### Variability

How stable is the result, where assessed?

### Error Analysis

What important failures occurred?

### Limitations

What conclusions cannot be supported?

### Decision

What should happen next?

---

# 48. Metric Registry

The following registry will evolve with the project.

| Research Dimension | Current / Candidate Metrics |
|---|---|
| Classification | Accuracy, sensitivity, specificity, precision, F1 |
| Discrimination | AUROC |
| Calibration | ECE, Brier score, reliability diagrams, NLL where appropriate |
| Confidence failure | High-confidence error analysis |
| Uncertainty | Predictive entropy and method-specific measures |
| Error detection | Error-detection AUROC, AUPRC |
| Selective prediction | Coverage, selective risk, risk-coverage curve |
| Distribution shift | Relative metric degradation + uncertainty response |
| Robustness | Performance across perturbation severity |
| Multimodal | Predictive + trustworthiness metrics + ablation |
| Generative | Grounding, factuality, unsupported generation, task-specific evaluation |

Metrics should be added only when they answer a meaningful research question.

---

# 49. What the Current Evidence Does Not Establish

Current experiments do not establish:

- clinical effectiveness;
- clinical safety;
- generalisation across hospitals;
- generalisation across patient populations;
- robustness to real-world acquisition differences;
- reliable uncertainty quantification;
- effective referral behaviour;
- multimodal reliability;
- generative AI reliability; or
- suitability for clinical deployment.

These remain research questions rather than assumed capabilities.

---

# 50. Governing Evaluation Principle

The project ultimately evaluates more than:

> **Was the prediction correct?**

It asks:

> **Was the prediction correct?**

> **Was its probability meaningful?**

> **Did the system recognise uncertainty?**

> **Could uncertainty help identify failure?**

> **Did reliability survive changing conditions?**

> **Could the system appropriately withhold or flag uncertain output?**

This leads to the governing principle:

> **Trustworthy healthcare AI requires evaluation of both predictive capability and the system's ability to recognise the limits of that capability.**
