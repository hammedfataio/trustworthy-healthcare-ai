# Trustworthy Healthcare AI

> **When a medical AI model reports 99% confidence, how trustworthy is that confidence?**

A research portfolio investigating **calibration, predictive uncertainty,
robustness and reliability in healthcare AI**.

The project progressively examines whether machine-learning systems can
recognise when their predictions should not be trusted.

---

## 🔬 Research Progress

| Stage | Research Question | Status |
|---|---|---|
| 001 — Baseline | How well does the baseline model perform? | ✅ Complete |
| 002 — Calibration | Can we trust its predicted probabilities? | ✅ Complete |
| 003 — Uncertainty | Can uncertainty help identify risky predictions? | 🔬 Next |
| 004 — Distribution Shift | What happens when the data changes? | Planned |
| 005 — Robustness | Does reliability survive challenging conditions? | Planned |

---

## 📊 Baseline Result

Using PneumoniaMNIST:

| Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| AUROC | 0.9370 |
| Sensitivity | 98.46% |
| Specificity | 71.79% |
| F1-score | 91.43% |

Despite strong aggregate performance, the model produced an incorrect
prediction with approximately **99.98% confidence**.

That failure became the motivation for the next stage of the research:
**probability calibration and uncertainty.**

---

## 🧪 Current Finding

Experiment 002 investigated probability calibration using reliability
analysis, Brier score, ECE and temperature scaling.

The learned temperature was:

`T = 1.007948`

Temperature scaling produced negligible improvement, suggesting that
calibration alone does not resolve the broader question of predictive
uncertainty.

This motivates **Experiment 003: Uncertainty Quantification**.

---

## 📚 Research Documentation

For the full research reasoning, methodology and experimental progression:

- [Research Log](docs/research_log.md)
- [Experiment 002 — Calibration](docs/experiment_002_calibration.md)

---

## 🧭 Research Direction

Baseline Classification  
↓  
Probability Calibration  
↓  
**Uncertainty Quantification**  
↓  
Distribution Shift  
↓  
Robustness  
↓  
Multimodal Clinical AI  
↓  
**Trustworthy Generative AI for Healthcare**

---

## ⚠️ Research Disclaimer

This repository contains experimental research for educational and research
purposes. It is not a medical device and should not be used for clinical
diagnosis.
