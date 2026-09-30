# Notebook Index

Read order and purpose of every notebook. This folder lives at `src/notebooks/`.
Notebooks `01`–`10` run at this top level and use relative paths like `../../data/`,
`../../results/` (up through `src/` to the repo root) — do **not** move them into
subfolders. Notebooks inside a subfolder (e.g. `11_integrated_gradients_bilstm/`)
go one level deeper again. The pivot work (patient-level) lives in its own folders.

---

## Approach 1 — Visit-level models (the baseline)
One row per admission (406,031 rows). Predicts readmit yes/no at 30/60/90 days.

| Notebook | Purpose |
|---|---|
| `01_data_loading_and_split.ipynb` | Load cohort, build train/test split |
| `02_exploratory_analysis.ipynb` | EDA, target and feature distributions |
| `03_baseline_models.ipynb` | Logistic-regression baselines |
| `04_xgboost_tuning.ipynb` | XGBoost tuning |
| `05_lstm_attention/` | Attention-LSTM (visit-level) |
| `06_lightgbm.ipynb` | LightGBM |
| `08_fairness_mitigation.ipynb` | FairGBM constrained variant, with/without-race check |
| `10_final_xgb.ipynb` | Final visit-level XGBoost |
| `18_visit_baseline_metrics/` | **Full-metrics rerun** (accuracy/precision/recall/F1/AUROC) — Roantree's ask |

## Explainability (XAI) — applies to the models above
| Notebook | Purpose |
|---|---|
| `07_shap_analysis/` | TreeSHAP + KernelSHAP |
| `09_lime_analysis.ipynb` | LIME |
| `11_integrated_gradients_bilstm/` | Integrated Gradients (BiLSTM) |
| `12_trust_framework/` | Trust framework (fidelity/plausibility/disagreement) |
| `13_trust_framework_figures/` | Trust scorecard, fidelity, disagreement-heatmap figures (visualizes `12`) |
| `14_tsne_patient_maps.ipynb` | t-SNE patient maps |

## Approach 2 — Patient-level pivot (the contribution)
One row per patient (180,352). Predicts how soon they return.

| Notebook | Purpose |
|---|---|
| `15_patient_dataset/` | Build patient-level dataset (`patient_first_visit.csv`, `patient_sequences.csv`) |
| `16_patient_models/` | Patient models: `16b_within_window_models` (30/60/90) + 5-bucket/regression |
| `17_lstm_incremental/` | LSTM incremental history (run in Colab / GPU) |
| `19_patient_shap/` | SHAP on the patient-level model |
| `20_fairness_audit/` | Fairness/race audit + clean race ablation |
| `21_survival_analysis/` | Survival analysis |

---

## Where outputs and support files live
- `data/processed/` — cleaned CSVs + the two patient-level files from `15_patient_dataset`
- `src/results/` — figures/CSVs; `src/results/NB12_trust_framework/` holds the trust-framework outputs
- `models/` — saved model artifacts
- `src/sql/` — cohort definition and feature-engineering queries
- `src/readmission/` — reusable package (data / model / explain)
- `docs/` — `ethics.pdf`, `notebook_design_notes.md`, proposal, progress reports, final paper

## Reading path for a new reader
`01` → `02` → Approach-1 models (`03`,`04`,`05`,`06`,`08`,`10`) → XAI (`07`,`09`,`11`,`12`) →
`15_patient_dataset` → `16_patient_models` → `17_lstm_incremental` → `19_patient_shap` → `20_fairness_audit` → `21_survival_analysis`.
