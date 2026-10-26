# Survivor Detection: Maritime Survival Prediction

Predict whether a passenger survived (1) or perished (0) in a major maritime incident from noisy tabular data. A Kaggle competition prepared for the IIT Madras BS in Data Science and Applications program, scored by accuracy. Solved with scikit-learn, XGBoost and CatBoost.

## Problem

The task is binary classification. The dataset was deliberately made "dirty" to simulate industry data: Gaussian noise, missing values and obfuscated features.

| | |
|---|---|
| Train | 712 passengers (`maritime_train.csv`) |
| Test | 179 passengers (`maritime_test.csv`) |
| Target | `Outcome` (1 = survived, 0 = deceased), about 38% positive |
| Metric | Categorisation accuracy: (TP + TN) / (TP + TN + FP + FN) |
| Rules | 1 member per team, 5 submissions per day, no external lookup of original survival results |
| Submission | `PassengerName`, `Outcome` |

Columns: `TicketTier` (1 to 3, a proxy for socio-economic status), `Gender`, `Age`, `RelativesAboard`, `ParentsChildren`, `TicketCost` (with artificial noise), `BoardingPort` (C, Q or S), `Class` (raw ticket string), `Berth`, `Singleton`, `FarePerPerson`, `Title`, and the name used as the identifier.

Data: [Survivor_Detection competition on Kaggle](https://www.kaggle.com/competitions/survival-detection/) (Jan 16 to Feb 5, 2026). The files are not included in this repo.

## Approach

1. **Cleaning**
   - Dropped `Berth` (78% missing).
   - Turned impossible negative ages into missing values.
   - Imputed age with the median of each **Title x Gender** group.
   - Filled the two missing ports with the mode.
   - Rounded the noisy `TicketTier` to integers and log-transformed `TicketCost`.
2. **Feature engineering**
   - Interaction features `RichWoman` (female, upper tier) and `PoorMan` (male, lower tier), plus `Gender_Tier`.
   - **`FamilySurvivalRate`**: mean survival of other passengers with the same surname, computed **out of fold** with `StratifiedKFold`, so a passenger's own label never enters their feature. Unseen families fall back to the global mean.
3. **Feature selection.** Dropped `RelativesAboard` and `ParentsChildren` (redundant with `FamilySize`) and `FarePerPerson` (lower mutual information than `TicketCost`).
4. **Preprocessing.** One-hot encoding for categories and standard scaling for numbers in a `ColumnTransformer`.
5. **Models.** Logistic Regression, KNN, Random Forest, XGBoost and CatBoost, each tuned with `GridSearchCV` and 5-fold cross-validation. The submitted model is the tuned CatBoost.

## Results

5-fold cross-validated accuracy after tuning:

| Model | CV accuracy |
|---|---|
| Random Forest | 0.840 |
| XGBoost | 0.840 |
| CatBoost (submitted) | 0.836 |
| Logistic Regression | 0.833 |
| KNN | 0.824 |

Same CatBoost model and notebook on the Kaggle leaderboard:

| Split | Accuracy |
|---|---|
| 5-fold cross-validation | about 83.6% |
| Public leaderboard | **86.58%** |
| Private leaderboard | **83.5%** |

The private score (the full hidden test set) agrees with cross-validation, so about 83 to 84% is the realistic accuracy. The public score is computed on only a portion of the 179-passenger test set, so a few lucky predictions can move it by a point or more.

With 712 rows the standard error of these scores is around 1 to 2 points, so the models are statistically very close. CatBoost was kept for stability on a small dataset with a strong prior for tabular data, not because it is provably better.

## Data leakage lesson

The family survival rate is a strong feature because families tended to live or die together. Computed on all training rows it contains each passenger's own label, which inflates training performance and does not generalise. Computing it from the **other folds only** removes that leak. This is the main point of the project: **a feature that looks great may be using the answer**.

## Limitations

- The scaler and one-hot encoder were fitted on the whole training set before cross-validation, and the folds used for `FamilySurvivalRate` differ from the folds used for model selection. Both add a small amount of optimism. The cleanest version puts all preprocessing and the encoding in one `Pipeline` with shared folds.
- CatBoost received the one-hot encoded matrix, so its native categorical handling was not used.
- Large grid searches on 712 rows overfit the model selection a little; repeated stratified CV would give a steadier estimate.
- Age is imputed with group medians but there is no "age was missing" flag.
- Accuracy is the competition metric; precision, recall, F1 and ROC-AUC would give a fuller picture.

## Future scope

Ensembles of the tuned models, a neural-network baseline, and serving the model behind an API.

## Run it

The notebook reads from `/kaggle/input/survival-detection/`, so the simplest way to run it is on Kaggle. To run it locally, download the three CSV files and change the paths in the data-loading cells.

```bash
pip install -r requirements.txt
jupyter lab survival-prediction.ipynb
```

## Files

| File | Purpose |
|---|---|
| `survival-prediction.ipynb` | Full notebook with outputs |
| `Building a High-Accuracy Survival Prediction System--final.pptx` | Project presentation slides |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT |

## License

[MIT](LICENSE). The competition data remains subject to Kaggle's rules.
