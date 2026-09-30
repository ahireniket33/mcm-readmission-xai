# Predicting and Explaining Hospital Readmission (MIMIC-IV)

MSc Computing (Data Analytics) project, Dublin City University — trust-aware explainable AI and fairness for hospital readmission models on MIMIC-IV.

## What's here
- **Approach 1 — visit-level models** (406,031 admissions; 30/60/90-day readmission): logistic regression, XGBoost, LightGBM, attention-LSTM, FairGBM.
- **Explainability**: TreeSHAP, KernelSHAP, LIME, Integrated Gradients (BiLSTM), plus a **trust framework** scoring explanations on fidelity, plausibility and inter-method disagreement across horizons.
- **Approach 2 — patient-level pivot** (180,352 patients): patient-level XGBoost, incremental-history LSTM, patient SHAP, fairness audit by race, and survival analysis (Cox, Kaplan–Meier).

See [`INDEX.md`](INDEX.md) for every notebook's purpose and the suggested reading order, and [`NB11_framework_spec.md`](NB11_framework_spec.md) for the trust-framework specification.

## Stack
Python, pandas, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, SHAP, LIME, lifelines, BigQuery (cohort extraction).

## Data
This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/), which requires credentialed PhysioNet access and a signed data use agreement. **No MIMIC data is included in this repository.** Notebook outputs that displayed patient-level rows have been cleared, and per-patient attribution arrays and trained model weights are excluded. To reproduce, obtain MIMIC-IV access and build the cohort locally.

## Author
Niket Ahire — [LinkedIn](https://www.linkedin.com/in/niket-ahire-512178291)
