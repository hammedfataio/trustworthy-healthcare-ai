# Trustworthy Healthcare AI

> **When a medical AI model reports 99% confidence, how trustworthy is that confidence?**

A research portfolio investigating **reliability, probability calibration, uncertainty, and robustness in healthcare AI**.

The project moves beyond predictive accuracy to investigate a more important question for high-stakes artificial intelligence:

> **Can we identify when a machine-learning model should — and should not — be trusted?**

---

## 🔬 Research Progress

| Experiment | Research Question | Status |
|---|---|---|
| 001 — Baseline Classification | How well does the baseline model perform? | ✅ Complete |
| 002 — Probability Calibration | Can we trust the model's predicted probabilities? | ✅ Complete |
| 003 — Uncertainty Quantification | Can uncertainty help identify risky predictions? | 🔬 Next |
| 004 — Distribution Shift | What happens when the data distribution changes? | 📋 Planned |
| 005 — Robustness | Does model reliability survive challenging conditions? | 📋 Planned |

---

## 📊 Baseline Findings

The baseline experiment established strong overall classification performance:

| Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| AUROC | 0.9370 |
| F1-score | 91.43% |
| Sensitivity | 98.46% |
| Specificity | 71.79% |

However, aggregate performance did not tell the whole story.

The model produced an **incorrect prediction with approximately 99.98% confidence**.

This high-confidence failure motivated the next stage of the research:

> **Can the model's confidence actually be trusted?**

---

## 🧪 Experiment 002 — Probability Calibration

Experiment 002 investigated model confidence using probability calibration and **temperature scaling**.

The learned temperature was:

**T = `1.007948`**

Validation Negative Log-Likelihood changed from:

| Before | After |
|---:|---:|
| 0.097432 | 0.097427 |

Classification performance remained unchanged:

- Accuracy: `0.8846 → 0.8846`
- AUROC: `0.9370 → 0.9370`
- F1-score: `0.9143 → 0.9143`
- Changed class predictions: `0`

The small calibration adjustment suggests that **global probability calibration alone does not resolve the broader trustworthiness problem**.

This motivates the next question:

> **Can prediction-level uncertainty provide information that confidence alone cannot?**

📄 [Read the full Experiment 002 report](docs/experiment_002_calibration.md)

---

## 🧠 Research Direction

The project follows a progressive experimental roadmap:

**Predictive Performance**  
↓  
**Probability Calibration**  
↓  
**Predictive Uncertainty** ← *Next*  
↓  
**Distribution Shift**  
↓  
**Robustness**  
↓  
**Trustworthy AI for Healthcare**

Each experiment builds on evidence from the previous stage rather than treating trustworthiness as a single metric.

---

## 📚 Research Documentation

Detailed research reasoning and experimental records are maintained separately from this README.

| Document | Purpose |
|---|---|
| [Research Log](docs/research_log.md) | Research reasoning, observations, and project progression |
| [Experiment 002 — Probability Calibration](docs/experiment_002_calibration.md) | Calibration methodology, results, interpretation, and limitations |

Additional experiment reports will be added as the research progresses.

---

## 📁 Repository Structure

```text
trustworthy-healthcare-ai/
├── docs/          # Research documentation and experiment reports
├── experiments/   # Experiment workflows/configurations
├── notebooks/     # Analysis and experimental notebooks
├── results/       # Experimental outputs
├── src/           # Reusable source code
├── tests/         # Automated tests
├── pyproject.toml # Python project configuration
└── uv.lock        # Locked dependency environment
