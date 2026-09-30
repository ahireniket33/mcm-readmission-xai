# Notebook 11 - Clinical Validation Framework (Spec)

Design notes for NB11. We build from this so we don't rework mid-build.

- Builds on: NB07 (SHAP), NB09 (LIME), 09.1_final_xgb (model + split).
- Feeds: NB10 (IG), NB12 (t-SNE), and the Methodology + Results sections of the paper.
- Last updated: 2026-06-07.

## 1. Why we are doing this

- Our test AUROC is 0.7183. Readmission state-of-the-art on richer data is ~0.78-0.81. On a single-site MIMIC-IV extract that gap is a data ceiling, not a modelling failure. We are not going to win on the metric.
- So the contribution is a framework, not a score. We ask: when SHAP, LIME and IG disagree about why a patient gets readmitted, which explanation should a clinician trust?
- We answer it on three axes: fidelity (does the explanation match the model), plausibility (does it match clinical literature), and disagreement (do the methods agree with each other).
- The novelty is the adjudication: using fidelity + plausibility to decide which method to trust when they disagree, across three horizons. That is where the marks sit (Contribution 30 + Technical 20).
- Nothing in this notebook competes on AUROC.

## 2. Method choice (read first)

We run the core framework on the XGBoost model with three tree-valid methods:

- TreeSHAP - exact Shapley values for trees (NB07 already uses it).
- KernelSHAP - model-agnostic Shapley estimator, black-box via predict_proba.
- LIME - local linear surrogate (NB09 already uses it).

We do NOT put Integrated Gradients in the core matrix:

- IG needs gradients. XGBoost is not differentiable. Running IG on a tree (or on a surrogate and calling it the tree's explanation) is something an examiner can take apart, and we can't risk that on the own-work gate.
- Instead we run IG on the BiLSTM-attention model (NB05), which is differentiable, and report it separately as a cross-architecture check (section 7). This still honours Crane's IG request, but only where IG is valid.

On the two SHAP methods looking similar:

- TreeSHAP and KernelSHAP estimate the same thing, so they should mostly agree. We use that on purpose: their agreement is our check that the pipeline is wired right. Once that holds, LIME's disagreement is the real, method-level disagreement worth reporting.

Viva line: IG needs gradients, so we don't force it onto a tree. We ran the tree framework on three tree-valid methods, used the two Shapley estimators as a correctness check, and applied IG to the BiLSTM where it actually works.

## 3. Data and artifact contract

Every axis uses the same inputs on the same patients. Disagreement and plausibility only compare if all methods explain identical rows.

Model:

- Explain the base booster (xgboost_final_45f.pkl), not the calibrated wrapper. TreeExplainer can't read a CalibratedClassifierCV. Calibration changes the probability scale, not which features drive the prediction, so explaining the base model is correct.
- Features: 45, from final_feature_cols.pkl.
- Split: final_split_indices.pkl (chronological, leakage-guarded).

Encoding:

- Encode categoricals by loading final_le_mappings.pkl. Do not re-fit a fresh LabelEncoder. NB07 re-fits one and gets away with it (LabelEncoder is deterministic and the AUROC reproduces), but re-fitting can silently mis-map categories. Load the saved mappings so SHAP, KernelSHAP, LIME and the reference set agree on category codes.

Shared sample (X_eval):

- X_eval = X_test.sample(2000, random_state=42), the exact sample NB07 uses for SHAP. Reuse it so SHAP isn't recomputed and all methods line up.
- Save the index to results/NB11/eval_index.npy so every method, horizon, and NB10/NB12 reload the same rows.
- Keep y_eval and p_eval = model.predict_proba(X_eval)[:,1] for the Quantus metrics and risk stratification.

Leakage columns (never in X):

- admittime, dischtime, days_to_next_admission, readmitted_30d/60d/90d. The readmitted_* columns are labels only (y, never X). admittime is used only to sort.
- Restate the guard in the header cell: assert test_min_date > train_max_date.

Attribution contract:

- get_attributions(method, model, X_eval) returns shape (2000, 45), signed attribution per feature, in final_feature_cols order. All three core methods return this shape so the rest of the code is method-agnostic.
- NB10's IG returns the same (n, d) shape but on the BiLSTM's own feature space (section 7), not in the tree matrix.

## 4. Axis A - Fidelity (Quantus)

Question: does the attribution actually reflect what the model does. An explanation can look clinically sensible and still be unfaithful.

- Library: quantus (Hedstrom et al., JMLR 2023). Wrap XGBoost as a callable around predict_proba. The metrics we use need only forward passes, no gradients, so XGBoost is fine.

Metrics (one or two per category):

- Faithfulness: Faithfulness Correlation (Bhatt 2020); Monotonicity-Correlation (Arya 2019).
- Robustness: Max-Sensitivity / Avg-Sensitivity (Yeh 2019).
- Complexity: Sparseness (Chalasani 2020) - concentrated attribution is easier to use at the bedside.

Randomisation control:

- The standard Quantus sanity check (Model Parameter Randomisation, Adebayo 2018) randomises model weights layer by layer. Trees have no layers, so it doesn't apply.
- We substitute a random-attribution baseline: score random attributions on the faithfulness metrics. A real method must beat random by a clear margin; if not, the attributions are noise. Report the gap (method minus random).

Output:

- Methods x metrics table (TreeSHAP, KernelSHAP, LIME) x (4 metrics + random gap), mean +/- std over the 2000 rows. Save to results/NB11/fidelity_h{H}.csv.

## 5. Axis B - Plausibility and the clinical reference set

Question: does the explanation point at features the literature says should drive readmission. This is where domain validity comes in.

Reference set:

- File: results/NB11/clinical_reference_set.json (done).
- Each of the 45 features has: tier (high / moderate / not_expected), expected_sign (positive / negative / null), a rationale, and a citation.
- The reference set = all high-tier features (currently 12). This is an evidence result, not a chosen number. If the literature supported 9, it would be 9. We defend each feature by its citation, never the count.
- Anti-circularity: tiers come from literature and clinical reasoning only, never from this model's SHAP output. We built it before looking at NB11 results. Otherwise plausibility just measures the model agreeing with itself.
- Clinician caveat: neither of us is a clinician, so this is a literature-grounded prior, not expert truth. Ten moderate features rest on general clinical knowledge with no peer-reviewed citation (flagged in the file). Plan to run it past Minh-Khoi Pham for a sanity check.

Scoring (we reuse Krishna's agreement measures, method vs reference instead of method vs method):

- Feature Agreement - overlap of each method's top features with the high-tier set.
- Sign Agreement - does the attribution sign match expected_sign.
- Rank Correlation - Spearman between method ranking and reference ranking over shared features.
- K is the read-depth into each method's ranking, separate from the 12. Report at K = 5, 10, 15 (same grid as section 6 [disagreement]). A range avoids defending one magic cutoff. 5 = quick glance, 10 = thorough read, 15 = generous upper bound.

Output:

- Method x measure table, results/NB11/plausibility_h{H}.csv. Optionally split by predicted-risk band (high vs low risk patients may get more or less plausible explanations).

## 6. Axis C - Cross-method disagreement (Krishna)

Question: do the methods agree with each other. If two methods give a clinician opposite stories for the same patient, that's a trust problem.

- Reference: Krishna et al., TMLR 2024 (peer-reviewed version, not the 2022 preprint - rubric bans non-peer-reviewed).

Measures (per patient, then averaged over the 2000):

- Feature Agreement, Rank Agreement, Sign Agreement, Signed-Rank Agreement, Rank Correlation, Pairwise Rank Agreement.

How:

- Pairwise: TreeSHAP-KernelSHAP, TreeSHAP-LIME, KernelSHAP-LIME.
- At K = 5, 10, 15. Report mean +/- std.

Control and adjudication:

- TreeSHAP-KernelSHAP is the positive control. It should be high; if it is, the pipeline works and TreeSHAP-LIME / KernelSHAP-LIME carry the real signal.
- Adjudication is the point: where methods disagree, look at Axis A (which is more faithful) and Axis B (which is more plausible) to say which explanation to trust. Produce a per-method trust scorecard combining the three axes. Disagreement is the question; fidelity and plausibility are how we answer it.

Output:

- Disagreement matrix (pair x measure x K), results/NB11/disagreement_h{H}.csv, plus a heatmap and the trust scorecard.

## 7. Horizon dial (30 / 60 / 90)

One parameter, HORIZON in {30, 60, 90}.

- For each H: relabel y = readmitted_{H}d (all three columns already exist), retrain XGBoost with the same pipeline / hyperparameters / chronological split, regenerate the three methods on the same X_eval rows, recompute all three axes.
- Build H=30 fully first, then 60 and 90 are config flips. NB07 cell 13 already prototypes the 30/60/90 SHAP comparison, so the approach is proven - reuse that pattern.
- Frame as explanation stability across horizons.
- Confound: positive rate rises as the window widens, so class balance shifts. Report base rates per horizon and keep threshold choices horizon-aware, so a score shift isn't misread as an explanation change.

## 8. IG cross-architecture check (separate section)

- Model: BiLSTM-attention (NB05, lstm_with_race.keras), which is differentiable.
- Cohort: Crane (11 Mar) said the LSTM cohort must use patients with >=2 admissions only. Confirm the BiLSTM honoured this before using its IG output.
- Method: Integrated Gradients (Captum or tf gradient tape), returning the (n, d) contract on the BiLSTM's feature space.
- Report IG fidelity and plausibility (against the same reference set). Do NOT put IG in the tree disagreement matrix - different feature spaces. Instead report cross-architecture concordance: does the sequence model use the same clinical features as the tree model. Agreement = the clinical signal is robust across architectures; disagreement = it's architecture-dependent, which is itself a finding.

## 9. Outputs (results/NB11/)

- eval_index.npy - the fixed 2000-row index.
- clinical_reference_set.json - done.
- fidelity_h{30,60,90}.csv
- plausibility_h{30,60,90}.csv
- disagreement_h{30,60,90}.csv
- trust_scorecard_h{30,60,90}.csv / .png
- ig_concordance.csv
- Figures: disagreement heatmaps, horizon-stability plots, trust scorecard.

## 10. How this maps to the rubric

- Contribution (30): the adjudication framework, reusing Krishna's measures as a plausibility-vs-clinical check, and explanation stability across horizons. Answers Crane's repeated "not enough novelty" note.
- Technical (20): combining fidelity + plausibility + disagreement across three methods and three horizons, with the evaluation evidenced (random baseline, positive control, base-rate handling) not apologised for.
- Paper quality (20): every reference-set feature cites the verified corpus; methods cited at peer-reviewed venues (SHAP = Lundberg & Lee NeurIPS 2017; LIME = Ribeiro KDD 2016; IG = Sundararajan ICML 2017; Quantus = Hedstrom JMLR 2023; Krishna = TMLR 2024). No arXiv.
- Ownership (5): we build and commit NB11 to GitLab ourselves.

## 11. Open items for Roantree (17 June)

- Reference set: accept the literature-grounded prior, or push for MKP / expert sign-off?
- Length of stay as a secondary outcome (Crane 22 Dec) - still not done. Add it or scope it out with a reason.
- 5-year recency rule: confirm the classic exception for SHAP/LIME/IG (all older than 5 years).
- 0.79 -> 0.72: Crane's file shows 0.79; one sentence explaining the leakage-corrected drop.
- UNKNOWN demographic handling in the with/without-race slice.

## 12. Build order

1. Section 3 contract cell: load model, assert split integrity, load/save X_eval index, define get_attributions, encode via saved le_mappings.
2. Reference set - done (results/NB11/clinical_reference_set.json).
3. Disagreement on H=30 - check the TreeSHAP-KernelSHAP control before trusting anything.
4. Fidelity on H=30 (with random baseline).
5. Plausibility on H=30.
6. Trust scorecard / adjudication on H=30.
7. Parameterise, run H=60 and H=90.
8. IG-on-BiLSTM concordance.
9. Save outputs, commit to GitLab.

## 13. Viva answers (plain)

We keep these short so we can say them in our own words.

Why explain the plain XGBoost, not the calibrated one:

- XGBoost does two jobs: rank patients, and turn the rank into an honest percentage. Calibration only fixes the second. It doesn't change which features drove the prediction, and TreeExplainer can't read the calibrated wrapper anyway. SHAP is about the reasons, so we explain the model before calibration. Example: whether a patient reads 31% or a calibrated 27%, the reason is the same - high num_admissions_last_90d.

Why a literature reference set is fine without a doctor:

- It's a literature-grounded prior, not a single doctor's opinion. Checking against published consensus is stronger and less biased than one clinician. Example: it flags language and marital_status as not-expected, so if LIME leans on a language code we catch it. We also say plainly that an expert review (MKP) would make it stronger.

Why TreeSHAP and KernelSHAP aren't "really one method":

- They estimate the same thing, so we use their agreement as a check that the experiment isn't broken. Once that holds, LIME (a different kind of method) disagreeing is the real result we set out to study. Example: if both SHAP methods top-rank num_diagnoses and num_medications but LIME puts los_days first, that's the moment we use fidelity and plausibility to decide which to trust.

Why the reference set is 12 features:

- We didn't pick 12. We tiered all 45 against the literature and 12 cleared the high-evidence bar. The number is a result. We defend each one with its citation, not the count.

What K we use and why:

- K is how deep we read into each method's ranking, not the size of the reference set. We report K = 5, 10, 15 instead of one value, so the result has to hold across depths. Picking one K is what we'd be attacked for; a range is the answer.
