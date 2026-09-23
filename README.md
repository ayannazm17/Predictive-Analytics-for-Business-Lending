# Predicting Business Credit Applications for Universal Plus

CRISP-DM analytics project identifying which existing customers are likely to apply for business credit, so relationship managers can engage proactively rather than reactively.

## Business problem

Universal Plus wants to grow business lending revenue and market share by identifying likely business credit applicants within its *existing* customer base, ahead of them applying — turning a reactive sales process into a proactive one.

## Data

100,000 anonymised customer records, 200 continuous numerical features (`var_0`–`var_199`), an `ID_code`, and a binary target (applied for business credit or not). All features anonymised — no domain knowledge available, so the whole workflow leans on statistical exploration rather than business-rule feature engineering.

Key data issues:
- `var_45`: ~40% missing
- Target imbalance: ~10/90 (positive/negative)
- No reliable way to distinguish genuine extreme values from data errors, given full anonymisation

## Method (CRISP-DM)

1. **Data understanding** — summary stats and distribution plots to characterise ranges, missingness and imbalance.
2. **Data preparation** — two parallel pipelines: Pipeline A drops `var_45`, Pipeline B keeps it (both impute remaining gaps with the median). Class imbalance handled via embedded model weighting (`class_weight="balanced"` / `scale_pos_weight`) rather than resampling, to avoid distorting the data. Feature reduction via Information Gain, keeping only positive-IG features. Stratified 80/20 train/test split.
3. **Modelling** — six classifiers compared: Logistic Regression, SVM (linear kernel, trained on a 20% subsample for tractability), Decision Tree, Random Forest, XGBoost, LightGBM.
4. **Evaluation** — AUC, recall, precision, F1, accuracy — with **recall prioritised**, since a missed applicant (false negative) is lost revenue, while the business can tolerate some wasted outreach.

Pipeline A (drop `var_45`) outperformed Pipeline B across the board and was used for all final model comparisons.

## Results

| Model | AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.854 | **0.771** | 0.272 | 0.402 | 0.770 |
| SVM (small dataset) | 0.850 | 0.728 | 0.268 | 0.392 | 0.778 |
| Decision Tree | 0.620 | 0.632 | 0.140 | 0.230 | 0.574 |
| Random Forest | 0.779 | 0.000 | 0.000 | 0.000 | 0.900 |
| XGBoost | 0.861 | 0.501 | 0.499 | 0.500 | 0.899 |
| LightGBM | 0.860 | 0.643 | 0.370 | 0.470 | 0.854 |

**Logistic Regression selected as the final model** — highest recall, strong AUC, and fully interpretable (a hard requirement for regulatory explainability and stakeholder trust in credit-adjacent decisions). Random Forest is the cautionary tale here: highest raw accuracy (0.90) but predicts *zero* positives — a textbook illustration of why accuracy is the wrong metric on an imbalanced target.

Confusion matrix for the final model: 1,549 true positives, 459 false negatives — the model catches most of the actual future applicants.

## Deployment plan

Model outputs feed relationship manager dashboards in the CRM, piloted on select branches/segments before full rollout, with ongoing monitoring and periodic recalibration as new customer data comes in.

## Limitations

Fully anonymised features mean no business interpretation of *why* the model flags a given customer — permutation importance ranks the top contributing variables, but without variable definitions this stays statistical rather than actionable insight. Access to real variable names would materially improve both performance and explainability.

## Repo contents

- `Group_10_Report.pdf` — full report, including literature review, all six model write-ups, ROC/precision-recall/cumulative gain comparison charts, confusion matrix and feature importance plots
