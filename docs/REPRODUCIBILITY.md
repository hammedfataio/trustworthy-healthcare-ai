# Reproducibility Framework

## Trustworthy Healthcare AI

**Environment, Experiment Traceability, Artifact Management, and Reproduction Standards**

---

## 1. Purpose

This document defines the reproducibility framework for the Trustworthy Healthcare AI project.

Reproducibility is treated as part of the research methodology rather than as a final documentation task.

The objective is to make it possible to trace a reported result from:

**research question → code → environment → data → model → evaluation → artifact → interpretation**

The governing principle is:

> **A research result is substantially more useful when the procedure that produced it can be identified, reconstructed, and independently inspected.**

---

## 2. Reproducibility Objectives

The project aims to make the following elements traceable:

- software environment;
- dependency versions;
- Python version;
- dataset source and split;
- preprocessing;
- model architecture;
- random seed;
- training configuration;
- evaluation procedure;
- model checkpoint;
- result tables;
- figures;
- experiment documentation; and
- Git history.

Not every experiment will require every artifact, but the information necessary to reproduce its principal result should be preserved.

---

## 3. Environment Management

The project uses:

**Python 3.11**

with:

**`uv`**

for Python environment and dependency management.

Important environment files include:

```text
.python-version
pyproject.toml
uv.lock
```

The project intentionally avoids relying on undocumented manual package installation.

---

## 4. Python Version

The repository pins the Python major/minor environment using:

```text
.python-version
```

Current target:

```text
3.11
```

The project configuration additionally constrains the supported Python version through `pyproject.toml`.

This reduces the risk of results depending on an unspecified interpreter environment.

---

## 5. Dependency Management

Project dependencies are declared in:

```text
pyproject.toml
```

The lockfile:

```text
uv.lock
```

records the resolved dependency environment.

Researchers reproducing the project should use the locked environment where possible rather than independently installing arbitrary package versions.

---

## 6. Environment Reconstruction

A typical environment reconstruction workflow is:

```bash
git clone <repository-url>
cd trustworthy-healthcare-ai
uv sync
```

The environment can then be used through `uv run`.

For example:

```bash
uv run python --version
```

or:

```bash
uv run jupyter lab
```

Exact execution commands may evolve as reusable scripts are introduced.

---

## 7. Repository Structure

The repository separates research code, notebooks, documentation, experiments, and outputs.

Conceptually:

```text
trustworthy-healthcare-ai/
│
├── docs/
│
├── experiments/
│
├── notebooks/
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

The structure may evolve as new validated capabilities are introduced.

Planned directories should not be represented as implemented functionality until they exist.

---

## 8. Dataset Provenance

Every dataset used in the project should be documented with sufficient provenance to identify:

- dataset name;
- source;
- version where available;
- task;
- modality;
- licence or usage restrictions where relevant;
- split procedure; and
- transformations.

The current benchmark is:

**PneumoniaMNIST**, from the MedMNIST benchmark collection.

The predefined dataset splits are retained for the initial experiments.

---

## 9. Current Dataset Splits

The current benchmark contains:

| Split | Samples |
|---|---:|
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |
| Total | 5,856 |

These roles must remain distinct.

```text
Training
   ↓
Model optimisation

Validation
   ↓
Method / parameter decisions

Test
   ↓
Held-out final evaluation
```

---

## 10. Data Transformation Traceability

Data transformations should be explicitly represented in code.

The initial experiments use image tensor conversion and normalisation appropriate to the baseline pipeline.

Future experiments should document changes such as:

- augmentation;
- resizing;
- normalisation;
- corruption;
- perturbation;
- modality alignment; or
- missing-data handling.

A preprocessing change should not occur silently between compared experiments.

---

## 11. Random Seeds

The initial experiments use:

```text
Seed = 42
```

Recording a seed improves repeatability but does not remove all possible sources of nondeterminism.

It also does not establish experimental stability.

Where stochastic variation may materially affect a conclusion, future experiments should consider multiple seeds.

---

## 12. Determinism

Machine-learning frameworks and hardware may introduce nondeterministic behaviour.

Where exact determinism is required and technically practical, deterministic settings should be documented.

Where exact reproducibility cannot be guaranteed, the limitation should be reported rather than concealed.

The project distinguishes:

**repeatability of procedure**

from

**bit-for-bit identity of every numerical output**.

---

## 13. Experimental Configuration

Every substantial experiment should preserve its important configuration.

Depending on the experiment, this may include:

```text
experiment_id
dataset
dataset_version
model
seed
batch_size
learning_rate
epochs
optimizer
loss_function
preprocessing
calibration_method
uncertainty_method
threshold_selection_method
evaluation_metrics
```

As the project matures, these settings may migrate into structured configuration files.

---

## 14. Model Traceability

Saved model artifacts should be associated with enough metadata to reconstruct their intended architecture and experimental context.

The EXP-001 checkpoint stores information including:

- model state;
- model name;
- dataset;
- input channels;
- number of outputs;
- image size;
- seed;
- batch size; and
- training epochs.

The baseline checkpoint is stored under:

```text
results/models/experiment_001_baseline_cnn.pt
```

---

## 15. Checkpoint Principle

A checkpoint should not be treated as self-explanatory.

A useful checkpoint requires surrounding context:

```text
Checkpoint
    +
Architecture Definition
    +
Dataset
    +
Preprocessing
    +
Configuration
    +
Experiment Documentation
```

Together these provide meaningful reproducibility.

---

## 16. Notebook Reproducibility

Notebooks are useful for research exploration but can create reproducibility problems if execution order is unclear.

Research notebooks should therefore aim to:

- use logical top-to-bottom execution;
- define imports explicitly;
- avoid hidden state;
- define random seeds;
- document dataset loading;
- display important configuration;
- preserve meaningful outputs;
- save important artifacts; and
- avoid depending on undocumented local files.

---

## 17. Notebook Naming

Current convention:

```text
01_baseline_medical_imaging.ipynb
02_confidence_calibration.ipynb
03_uncertainty_quantification.ipynb
...
```

The numeric prefix connects notebook progression with experiment progression.

Where practical:

```text
01 → EXP-001
02 → EXP-002
03 → EXP-003
```

---

## 18. Notebook-to-Source Transition

Notebooks are not intended to become the permanent home of all reusable functionality.

As code becomes stable, reusable components should migrate into:

```text
src/
```

For example:

```text
src/data/
src/models/
src/evaluation/
src/uncertainty/
```

This supports testing and reuse across experiments.

---

## 19. Result Artifact Standard

Important numerical results should not exist only as notebook output.

Where appropriate, experiments should save structured results under:

```text
results/tables/
```

Figures should be saved under:

```text
results/figures/
```

Model artifacts should be saved under:

```text
results/models/
```

This allows reported findings to be traced independently of notebook display state.

---

## 20. Result Naming

Artifact names should include the corresponding experiment identifier where practical.

Examples:

```text
experiment_001_baseline_metrics.csv

experiment_001_baseline_cnn.pt

experiment_002_calibration_comparison.csv

experiment_002_reliability_diagram.png

experiment_003_uncertainty_metrics.csv
```

This prevents ambiguous files such as:

```text
results.csv
final.csv
new_results.csv
model2.pt
```

---

## 21. Figure Reproducibility

Figures used in:

- experiment reports;
- GitHub documentation;
- technical posts; or
- future publications

should ideally be generated from version-controlled code and saved data.

Important plots should not depend on manual editing that cannot be reproduced.

---

## 22. Metrics Reproducibility

Metric implementations should be traceable.

Where established libraries are used, the relevant library should be identifiable from the environment.

Where custom implementations are required, such as a specific ECE calculation, the implementation should be preserved in code.

Important methodological choices should be documented.

For ECE, examples include:

- number of bins;
- binning strategy;
- treatment of boundary probabilities; and
- weighting procedure.

---

## 23. Calibration Reproducibility

Calibration experiments should preserve:

```text
Original model
      ↓
Validation predictions
      ↓
Calibration fitting
      ↓
Frozen calibration parameters
      ↓
Held-out evaluation
```

The test set should not be used to fit calibration parameters.

For EXP-002:

```text
Learned temperature = 1.007948

Validation NLL:
0.097432 → 0.097427
```

This procedure and its limitations should remain documented.

---

## 24. Threshold Reproducibility

Future experiments may introduce thresholds for:

- uncertainty;
- referral;
- abstention;
- risk categorisation; or
- shift detection.

Threshold selection should document:

- metric being optimised;
- data split used;
- selection rule; and
- final selected threshold.

A threshold should not appear in the final system without an identifiable derivation.

---

## 25. Distribution-Shift Reproducibility

Future distribution-shift experiments should preserve the exact shift-generation procedure.

For synthetic or controlled shifts, record:

```text
Shift Type
Shift Parameter
Shift Severity
Random Seed where applicable
Transformation Order
```

This enables the same shifted evaluation set to be reconstructed.

---

## 26. Robustness Reproducibility

Robustness experiments should avoid vague descriptions such as:

> Noise was added to the images.

Instead, the experiment should record:

- perturbation type;
- parameterisation;
- severity;
- application probability;
- order relative to preprocessing; and
- random seed where relevant.

---

## 27. Uncertainty Reproducibility

EXP-003 and later uncertainty experiments should document:

- uncertainty definition;
- mathematical quantity;
- model architecture;
- stochastic mechanism where applicable;
- number of stochastic passes where applicable;
- aggregation procedure;
- random seeds;
- error-detection metrics; and
- threshold-selection procedure.

Different uncertainty methods should not be treated as interchangeable.

---

## 28. Multi-Seed Experiments

Where multiple seeds are introduced, the project should preserve results per seed rather than only the final mean.

Conceptually:

```text
Seed 1 ─┐
Seed 2 ─┤
Seed 3 ─┼──► Aggregate
Seed 4 ─┤
Seed 5 ─┘
```

This permits later inspection of variability.

---

## 29. Statistical Reproducibility

Where confidence intervals or statistical comparisons are used, the project should document:

- statistical method;
- resampling procedure where applicable;
- number of bootstrap samples where applicable;
- confidence level;
- random seed; and
- comparison definition.

Statistical procedures should be reproducible from saved predictions or result artifacts where possible.

---

## 30. Prediction-Level Artifacts

For some trustworthiness experiments, aggregate metrics alone may be insufficient.

Where appropriate, prediction-level outputs may be preserved in a structured form containing non-sensitive research data such as:

```text
sample_identifier
true_label
predicted_label
predicted_probability
uncertainty
correctness
experimental_condition
```

This can allow later recalculation of:

- calibration;
- error detection;
- selective prediction;
- confidence intervals; and
- robustness analyses.

Sensitive healthcare information should not be included without appropriate governance.

---

## 31. Experiment Documentation

Every completed major experiment should have a corresponding report.

Current examples:

```text
docs/experiment_001_baseline.md
docs/experiment_002_calibration.md
```

Future example:

```text
docs/experiment_003_uncertainty.md
```

Each report should identify:

- research question;
- hypothesis;
- method;
- configuration;
- results;
- interpretation;
- limitations;
- artifacts; and
- next decision.

---

## 32. Research Log

The research log records chronological progress and reasoning.

Location:

```text
docs/research_log.md
```

It should capture important transitions such as:

```text
Observation
    ↓
Interpretation
    ↓
Decision
    ↓
Next Experiment
```

This complements formal experiment reports by preserving the evolution of the research.

---

## 33. Git as Research History

Git provides a version-controlled history of:

- code;
- documentation;
- configuration; and
- experimental progression.

Commits should describe meaningful research or engineering changes.

Preferred examples:

```text
Add baseline medical imaging experiment

Add probability calibration experiment

Add uncertainty quantification evaluation

Add distribution shift analysis
```

Avoid ambiguous messages such as:

```text
update

changes

final

fix stuff
```

---

## 34. Commit Scope

Where practical, commits should represent coherent units of work.

A research-stage commit may contain:

```text
Notebook
+
Result Table
+
Figure
+
Experiment Report
+
Research Log Update
```

This creates a useful historical checkpoint.

---

## 35. Documentation Traceability

The documentation hierarchy is:

```text
README.md
      ↓
RESEARCH_PLAN.md
      ↓
SYSTEM_ARCHITECTURE.md
      ↓
METHODOLOGY.md
      ↓
EXPERIMENTS.md
      ↓
EVALUATION.md
      ↓
REPRODUCIBILITY.md
      ↓
Individual Experiment Reports
      ↓
Notebooks + Results
```

Each level answers a different question.

### README

What is this project?

### Research Plan

What research programme is being pursued?

### System Architecture

What system is being built?

### Methodology

How is the research conducted?

### Experiments

What experiments provide the evidence?

### Evaluation

How is success or failure assessed?

### Reproducibility

Can the procedure and evidence be reconstructed?

---

## 36. Reproduction Checklist

Before an experiment is marked complete, ask:

```text
[ ] Is the research question documented?

[ ] Is the dataset identifiable?

[ ] Are data splits documented?

[ ] Is preprocessing documented?

[ ] Is the model architecture identifiable?

[ ] Are important hyperparameters recorded?

[ ] Is the random seed recorded?

[ ] Is the evaluation procedure documented?

[ ] Are thresholds derived without test leakage?

[ ] Are important metrics saved?

[ ] Are important figures saved?

[ ] Is the model artifact saved where necessary?

[ ] Are limitations documented?

[ ] Is the experiment report complete?

[ ] Is the research log updated?

[ ] Is the work version controlled?
```

Not every checkbox applies identically to every experiment, but unexplained omissions should be avoided.

---

## 37. Reproducing EXP-001

EXP-001 should be reconstructable from:

```text
Environment
    ↓
PneumoniaMNIST
    ↓
Baseline preprocessing
    ↓
BaselineCNN
    ↓
Training configuration
    ↓
Held-out evaluation
    ↓
Metrics
```

Primary artifacts include:

```text
notebooks/01_baseline_medical_imaging.ipynb

results/models/experiment_001_baseline_cnn.pt

results/tables/experiment_001_baseline_metrics.csv

docs/experiment_001_baseline.md
```

---

## 38. Reproducing EXP-002

EXP-002 depends on the EXP-001 model.

Conceptually:

```text
EXP-001 Checkpoint
       ↓
Validation Predictions
       ↓
Temperature Fitting
       ↓
Frozen Temperature
       ↓
Held-Out Test Predictions
       ↓
Calibration Evaluation
```

Primary documentation:

```text
notebooks/02_confidence_calibration.ipynb

docs/experiment_002_calibration.md
```

Associated result tables and figures should remain stored under `results/` when produced.

---

## 39. Reproducing EXP-003

EXP-003 should not begin without documenting the uncertainty method.

The reproducibility record should include:

```text
Baseline Model
      ↓
Uncertainty Definition
      ↓
Implementation
      ↓
Validation Procedure
      ↓
Error-Detection Evaluation
      ↓
Held-Out Results
```

If a new stochastic model is trained, its architecture and training configuration must be recorded separately from EXP-001.

---

## 40. External Reproducibility

Internal repeatability is only the first level of reproducibility.

As the project matures, the repository should aim to allow an independent researcher to:

1. clone the repository;
2. recreate the environment;
3. obtain the documented public dataset;
4. execute the relevant experiment;
5. reproduce the analysis procedure; and
6. compare generated results with reported evidence.

Differences caused by hardware, framework nondeterminism, or stochastic training should be acknowledged.

---

## 41. Computational Environment

Experiments should record relevant computational context when it materially affects reproduction.

Examples include:

- CPU vs GPU;
- accelerator type where relevant;
- operating environment;
- framework version; and
- major hardware-specific constraints.

The initial experiments were executed using CPU-based PyTorch.

Future experiments involving larger multimodal or generative models may require more detailed computational reporting.

---

## 42. Data Availability

The project currently uses a public benchmark rather than private patient data.

Future datasets should clearly document whether they are:

- publicly available;
- available under registration;
- restricted;
- institutionally governed; or
- non-shareable.

A reproducibility claim should never imply that restricted healthcare data can be redistributed.

---

## 43. Privacy Boundary

Reproducibility must not override patient privacy, ethics, or data governance.

If future research uses sensitive healthcare data, reproducibility may require sharing:

- code;
- configuration;
- synthetic examples;
- derived non-identifiable statistics; or
- instructions for authorised data access

rather than distributing protected records.

---

## 44. Model and Dataset Versioning

As the project expands, model and dataset versions should be identifiable.

A future naming strategy may include:

```text
model_version
dataset_version
experiment_id
configuration_id
```

This becomes increasingly important when comparing multiple models and datasets.

---

## 45. Reproducibility vs Replicability

Within this project:

**Reproducibility** refers primarily to reconstructing an analysis using the documented data, code, environment, and procedure.

**Replicability** refers more broadly to obtaining consistent scientific conclusions through an independently repeated study or related dataset.

The current repository primarily targets strong computational reproducibility.

Later external-dataset experiments can provide stronger evidence of replicability and generalisation.

---

## 46. Known Current Limitations

Current reproducibility limitations include:

- some research logic remains notebook-centred;
- the initial model is based on a single principal training seed;
- the current benchmark is small;
- hardware-independent numerical identity is not guaranteed;
- some experiment configuration is embedded directly in notebooks;
- automated end-to-end experiment execution is not yet implemented; and
- continuous integration for research tests is not yet fully established.

These are engineering and methodological opportunities for later stages.

---

## 47. Future Reproducibility Improvements

As the system matures, potential improvements include:

```text
Structured experiment configuration
        ↓
Reusable training pipeline
        ↓
Reusable evaluation pipeline
        ↓
Automated tests
        ↓
Continuous integration
        ↓
Experiment metadata
        ↓
Model/data version tracking
        ↓
Automated artifact generation
```

These should be introduced when they improve the research process rather than merely increase tooling complexity.

---

## 48. Definition of Reproducible Experiment

Within this project, an experiment is considered sufficiently reproducible when another technically competent researcher can determine:

> What data were used?

> How were the data processed?

> Which model was evaluated?

> How was it trained or loaded?

> Which parameters affected the experiment?

> How were validation and test data used?

> Which metrics produced the reported result?

> Where are the corresponding artifacts?

> What limitations affect reproduction?

If these questions cannot be answered from the repository, the reproducibility record is incomplete.

---

## 49. Reproducibility Principle

The project does not treat reproducibility as simply:

```text
"The code is on GitHub."
```

GitHub availability alone does not make research reproducible.

Instead:

```text
Code
+
Environment
+
Data Provenance
+
Configuration
+
Experimental Procedure
+
Artifacts
+
Documentation
+
Version History
        ↓
Reproducible Research Evidence
```

The governing principle is:

> **Every important research claim should be traceable to an identifiable experimental procedure and its corresponding evidence.**
