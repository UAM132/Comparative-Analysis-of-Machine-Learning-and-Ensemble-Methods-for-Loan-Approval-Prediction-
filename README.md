# Comparative Analysis of Machine Learning and Ensemble Methods for Loan Approval Prediction

Five classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, Neural Network) and three ensembles
(Majority Voting, Weighted Voting, Stacking) on the Kaggle Loan Approval Prediction Dataset (4,269 applicants).

## Run
```
pip install -r requirements.txt
python loan_pipeline_clean.py loan_approval_dataset.csv
```
Everything is written to `results/`. Random seed is fixed (42), so the numbers are reproducible.

## What the script does
- Stratified 80/10/10 train/validation/test split. Validation set is used only for the weighted-voting weights; the test set is used once.
- Five-fold stratified cross-validation of the whole pipeline -> `cv_folds_*.csv`, `cv_summary_*.csv` (mean ± std).
- Feature importance (Random Forest impurity + permutation importance for Random Forest and Stacking) -> `feature_importance.csv/.png`.
- Credit-score checks: one-feature baseline (`credit_score_only_baseline.csv`) and all eight models re-run without `cibil_score`
  (`test_without_credit_score.csv`, `cv_summary_without_credit_score.csv`).

## Output files
| File | Content |
|---|---|
| results_split.csv | Train / validation / test metrics for all 8 models |
| results_confusion.csv | Test-set confusion matrices |
| weighted_voting_weights.csv | Validation-precision weights |
| cv_folds_all_features.csv, cv_summary_all_features.csv | 5-fold CV, all features |
| feature_importance.csv, feature_importance.png | Feature importance |
| credit_score_only_baseline.csv | Single-feature baseline |
| test_without_credit_score.csv, cv_*_without_credit_score.csv | Ablation without credit score |
