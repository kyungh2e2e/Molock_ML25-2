# Closed- & Open-World Website Fingerprinting
**Team 07 Molock**

> 🔎 Machine Learning Team Project — **Closed-world baseline**, plus a **feature selection and modeling pipeline** for Closed- & Open-World evaluation.

This repository contains:

- A **closed-world Random Forest baseline** with simple handcrafted features.
- A **final model** that:
  - Extracts **37 statistical / burst-based features** from packet sequences.
  - Reduces them to **27** and then **26** features using:
    - Correlation-based pruning  
    - Zero-importance pruning  
    - IQR-based outlier removal  
    - Ablation study and small combinatorial search
  - Prepares models that can be evaluated in **closed-world** and **open-world** scenarios (binary detection, multi-class, and hierarchical two-stage).

---

## 🚀 Quick Start

### 0.1 Requirements

```bash
python >= 3.8
pip install -r requirements.txt
```

---

### 0.2 Dataset Setup

This project uses two types of datasets:

| Dataset | Used for | Included in Repo | File |
|---------|----------|------------------|------|
| **Raw traffic dataset** | Baseline / Feature Selection | ❌ (private & large) | `mon_standard.pkl`, `unmon_standard.pkl` |
| **Processed feature dataset (26 features)** | Final Modeling | ✔ Included | `processed_traffic_data_26feats.pkl` |

---

### Setup Guide

#### A) To run Baseline / Feature Selection notebooks
You must have access to the raw dataset:
```

mon_standard.pkl
unmon_standard.pkl

```
These files are not included in the repository due to privacy and size restrictions.

> If you have access, place them in the project root before running:
- `baseline_feature10.ipynb`
- `feature_selection_37_27_26.ipynb`

#### (B) To run Final Modeling notebook
No raw dataset required. Only the final processed file is needed.

```

processed_traffic_data_26feats.pkl

```

This file is already included.  
If running on Colab, upload it to the working directory.

---

### Dataset pipeline diagram

```
Raw Packet
   ↓ extract 37 features
Feature Selection → 27
Ablation / Tuning → 26   → final_modeling_code.ipynb
```

---

You can either:

#### (A) Run locally

Place the `.pkl` files under a `data/` directory and set paths in the notebooks, e.g.:

```python
MON_PATH = "data/mon_standard.pkl"
UNMON_PATH = "data/unmon_standard10.pkl"
```

#### (B) Run in Google Colab (with Google Drive)

Upload the dataset to your own Google Drive and mount it:

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/"  # <- modify this
MON_PATH = DATA_DIR + "mon_standard.pkl"
UNMON_PATH = DATA_DIR + "unmon_standard10.pkl"
```

If the dataset is not found, update `DATA_DIR` to match your own Drive path.

---

### 0.3 Minimal Run Order

#### Run locally

1. **Closed-world baseline**

   ```bash
   jupyter notebook baseline_feature10.ipynb
   ```

2. **Feature selection (37 → 27 → 26)**

   ```bash
   jupyter notebook final_code_featureSelection.ipynb
   ```

3. **Closed/Open-world modeling & scenarios**

   ```bash
   jupyter notebook final_modeling_code.ipynb
   ```

For detailed descriptions, see Section **8. How to Run (Detailed)**.

#### Run in Google Colab

Open the notebooks in Colab using the buttons below.

baseline_feature10.ipynb : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyungh2e2e/Molock_ML25-2/blob/main/baseline_feature10.ipynb)

final_code_featureSelection.ipynb : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyungh2e2e/Molock_ML25-2/blob/main/final_code_featureSelection.ipynb)

final_modeling_code.ipynb : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyungh2e2e/Molock_ML25-2/blob/main/final_modeling_code.ipynb)


---

## 1. Repository Structure

```text
.
├── baseline_feature10.ipynb          # 10-feature closed-world baseline
├── final_code_featureSelection.ipynb # 37→27→26 feature selection pipeline (closed-world)
├── final_modeling_code.ipynb         # Closed/Open-world modeling & scenario experiments
├── data/                             # dataset .pkl files (user-provided)
├── LICENSE                           # (optional) license file
└── README.md                         # this document
```

---

## 2. Data Description

### 2.1 Monitored Data (`mon_standard.pkl`)

* Contains **monitored website traces** preprocessed into sequences.

* Each trace is represented as:

  * `X1`: timestamp sequence
  * `X2`: signed size sequence (direction × fixed size, `±512`)

* Labels:

  * `USE_SUBLABEL = False` → website-level labels (e.g., 95 classes)
  * `URL_PER_SITE = 10`
  * `TOTAL_MON_URLS = 950` (95 sites × 10 URLs per site)

* Loaded via:

  ```python
  X1_mon, X2_mon, y_mon = load_monitored(
      mon_path=MON_PATH,
      use_sublabel=USE_SUBLABEL,
      url_per_site=URL_PER_SITE,
      total_urls=TOTAL_MON_URLS
  )
  ```

### 2.2 Unmonitored Data (`unmon_standard10.pkl`)

* Contains **unmonitored / background traffic**.
* Used primarily in the **open-world experiments** in `final_modeling_code.ipynb`.
* In open-world settings:

  * Monitored samples are labeled as **0–94**.
  * Unmonitored samples are labeled as **-1** (reject class).

---

## 3. Closed-World Baseline (10 Features)

The baseline (in `baseline_feature10.ipynb`) constructs a simple closed-world classifier using **10 manually engineered traffic features** extracted from `(X1, X2)`.

Conceptually, the baseline:

1. Loads monitored data only.

2. Extracts 10 features per trace, such as:

   * Duration
   * Inter-arrival time statistics
   * Cumulative bytes
   * Burst-level statistics
   * In/out packet counts and ratios
   * Outgoing inter-arrival variability

3. Trains a `RandomForestClassifier` with a fixed configuration (e.g., `n_estimators=200`).

4. Evaluates accuracy and feature importances.

5. Performs a small **leave-one-feature-out ablation** to see which features are more critical.

This closed-world baseline acts as a **simple, reasonably strong classical reference model**, not a tuned SOTA system.

---

## 4. Final Model – 37 Feature Extraction

The final model uses a richer set of **37 features** extracted from `(X1, X2)` sequences.

Given:

* `X1`: timestamps
* `X2`: signed sizes (direction × fixed size, ±512)

The extractor computes:

#### 4.1 Basic traffic statistics

* `TotalTime` — `X1[-1] - X1[0]` (session duration)
* `TotalPackets` — number of packets
* `TotalVolume` — sum of absolute packet sizes
* `NumIncoming`, `NumOutgoing` — packet counts by direction
* `IncomingRatio`, `OutgoingRatio` — direction ratios
* `InOutDiff` — `NumIncoming - NumOutgoing`
* `Global_Slope` — `TotalVolume / TotalTime` (if `TotalTime > 0`)
* `First30_InRatio` — ratio of incoming packets within the first 30 packets

#### 4.2 Inter-Packet Interval (IPI) features

Based on `ipi = diff(X1)`:

* `IPI_Mean`, `IPI_Std`, `IPI_Max`, `IPI_Min`
* `IPI_Skew`, `IPI_Kurt` (skewness, kurtosis)
* Quantiles: `IPI_P10`, `IPI_P25`, `IPI_P50`, `IPI_P75`, `IPI_P90`

#### 4.3 Packet size distribution features

On `abs_sizes = abs(X2)`:

* Quantiles: `Size_P10`, `Size_P25`, `Size_P50`, `Size_P75`, `Size_P90`

#### 4.4 Burst-level features

Bursts are defined as consecutive packets with the **same direction sign**. For each burst:

* `BurstLen_*`: length-based statistics

  * `BurstLen_Count`, `BurstLen_Mean`, `BurstLen_Std`, `BurstLen_Max`, `BurstLen_Min`
* `BurstVol_*`: volume-based statistics

  * `BurstVol_Mean`, `BurstVol_Std`, `BurstVol_Max`, `BurstVol_Min`

#### 4.5 Cumulative volume features

On the cumulative sum of absolute sizes:

* `CumSum_Mean`, `CumSum_Std`

In total:

```text
Total features: 37
```

These are wrapped into a DataFrame via:

```python
X_df = build_feature_df(X1_mon, X2_mon)
# X_df.shape ≈ (n_samples, 37)
```

---

## 5. Final Model – Feature Selection Pipeline (37 → 27 → 26)

This section describes the **full feature selection pipeline** implemented in `final_code_featureSelection.ipynb`.

### 5.1 Step 1 – Correlation Analysis & Base RF

1. Compute the **correlation matrix** of the 37 features.

2. Identify **highly correlated feature pairs** with `|r| ≥ 0.9` and store them in `high_corr_pairs`.

3. Train a baseline `RandomForestClassifier` on all 37 features:

   * Split the dataset once (`train_test_split` with `stratify=y`).
   * Fit RF (e.g., `n_estimators=200`).
   * Compute:

     * Baseline accuracy
     * Feature importances (`importances: Series[feature → importance]`)

4. Visualize:

   * Correlation heatmap (seaborn heatmap)
   * Feature importances bar plot

This gives:

* A **baseline performance reference**.
* An initial idea of **redundant** and **important** features.

---

### 5.2 Step 2 – Correlation-Based Pruning

Using `high_corr_pairs` and RF importances:

1. For each correlated pair `(a, b, r)` with `|r| ≥ 0.9`:

   * Compare `importances[a]` vs `importances[b]`.
   * Mark the **less important** feature as a drop candidate.

2. Remove these candidates from `X_df` to obtain a pruned DataFrame `X_pruned_corr`.

3. Evaluate accuracy using the **same fixed train/test split**:

   ```python
   base_acc_corr = rf_acc_with_indices(X_df, y_mon, idx_train_corr, idx_test_corr)
   pruned_acc_corr = rf_acc_with_indices(X_pruned_corr, y_mon, idx_train_corr, idx_test_corr)
   ```

4. Compare `pruned_acc_corr` to `base_acc_corr`.

In practice, **correlation-based pruning reduced the number of features but did not improve accuracy**.
In fact, performance slightly degraded, so **this correlation-only pruning strategy was not adopted in the final feature set**.

---

### 5.3 Step 3 – Zero-Importance Pruning (37 → 27)

Next, features with **near-zero RF importance** are removed.

1. Identify zero-importance (or effectively zero) features:

   ```python
   zero_importance_feats = importances[importances <= 1e-6].index.tolist()
   ```

2. Measure baseline accuracy with **all 37 features** on the same split.

3. Drop the zero-importance features:

   ```python
   X_imp_pruned = X_df.drop(columns=zero_importance_feats)
   acc_imp_pruned = rf_acc_with_indices(X_imp_pruned, y_mon, idx_train_corr, idx_test_corr)
   ```

4. Compare:

   * `accuracy (37 features)` vs `accuracy (37 − |zero_importance_feats|)`.

This yields a **27-feature pruned set**, where features with **essentially no contribution** (according to RF importance and ablation) are removed.
In our experiments, the 27-feature configuration slightly **improved or at least maintained accuracy** compared to the 37-feature baseline, so we **kept this 27-feature set as our main pruned feature configuration**.

---

### 5.4 Step 4 – IQR Outlier Removal & Accuracy Comparison

Goal: compare different combinations of **feature sets** (full vs pruned) and **outlier handling** (raw vs cleaned).

#### 5.4.1 IQR-based outlier cleaning

For a given DataFrame `df`, the IQR cleaner:

1. For each feature column:

   * Compute `Q1`, `Q3`, and `IQR = Q3 − Q1`.
   * Define lower/upper bounds:
     `lower = Q1 − 1.5 * IQR`, `upper = Q3 + 1.5 * IQR`.
   * Mark samples outside `[lower, upper]` as outliers.

2. Compute a global mask across all features.

3. Apply the mask to get `df_clean` and the corresponding `y_clean`.

This is applied to:

* `X_full` = all 37 features
* `X_pruned` = 27-feature pruned DataFrame

resulting in:

* `X_full_clean`, `X_pruned_clean`

#### 5.4.2 RF accuracy for four configurations

The code evaluates four setups:

1. `Full 37 / Raw`
2. `Full 37 / Clean`
3. `Pruned 27 / Raw`
4. `Pruned 27 / Clean`

For each:

* Train a RF with a **fixed configuration** and stratified split.
* Record:

  * Number of features
  * Number of samples after cleaning
  * Accuracy

The results are summarized in a small DataFrame and the **best config** is selected based on accuracy.

This step provides evidence for:

* Whether **IQR cleaning** helps.
* Whether the **pruned (27-feature) set** outperforms the full 37-feature set when combined with outlier removal.

---

### 5.5 Step 5 – Ablation on Pruned + Clean (27 Features)

The focus then shifts to `X_pruned_clean` (27 features after pruning and IQR cleaning).

Steps:

1. Fix a stratified train/test split (indices).

2. Train RF using all 27 features → get **baseline accuracy**.

3. For each feature `f`:

   * Drop `f` from the DataFrame.
   * Retrain RF with the same split.
   * Record:

     * Accuracy with `f` removed
     * `Δ accuracy = acc_after_drop − base_acc_clean`

4. Sort features by their ablation performance.

This ablation reveals:

* Which features are **critical** (large negative Δ when removed).
* Which features are **redundant** or less impactful.

These insights guide the next step, a **small combinatorial search**.

---

### 5.6 Step 6 – Searching for a Compact 26-Feature Set

Based on the ablation results, a subset of relatively weak features is selected, e.g.:

```python
to_drop = ["IPI_Mean", "BurstLen_Max", "BurstVol_Mean", "BurstLen_Std"]
```

Then the code:

1. Enumerates **all non-empty combinations** of these candidates.

2. For each combination:

   * Drops the corresponding features from `X_pruned_clean`.
   * Retrains RF with fixed split.
   * Measures accuracy.

3. Sorts all combinations by accuracy.

In our experiments, **one of these combinations produced a 26-feature configuration with the best accuracy among the tested subsets**.
This best-scoring 26-feature set was therefore **included as a candidate configuration** for further comparison and later modeling.

---

### 5.7 Step 7 – Fair Comparison: 37 vs 27 vs 26 (Common Clean Mask)

To compare the 37-, 27-, and 26-feature models fairly:

1. Start from the original 37-feature DataFrame `X_37`.

2. Apply **a single, common IQR cleaning pass** to `X_37`:

   ```python
   X_37_clean, common_mask = clean_outliers_iqr(X_37)
   y_clean = y_all[common_mask]
   ```

3. Derive:

   * `X_27_clean` by dropping the zero-importance 10 features from `X_37_clean`.
   * `X_26_clean` by further dropping `IPI_Mean` (or the chosen feature) from `X_27_clean`.

4. Train three RF models:

   * RF on 37 features
   * RF on 27 features
   * RF on 26 features

5. Use the **same cleaned subset and same split** for all three models.

6. Evaluate via `evaluate_model_performance(..., scenario="multi")`.

This produces a **fair, apples-to-apples comparison**:

* 37-feature baseline
* 27-feature pruned model
* 26-feature compact candidate

In practice, the 27- and 26-feature models **match or slightly outperform** the 37-feature baseline, justifying their use in the final modeling notebook.

---

## 6. Evaluation Utilities

A common evaluation function is defined to standardize reporting:

```python
evaluate_model_performance(
    model,
    X_test,
    y_test,
    model_name="...",
    scenario="binary" or "multi"
)
```

Features:

* Prints:

  * Accuracy
  * Precision, Recall, F1
  * Unmonitored recall (when label `-1` exists)
  * Macro-averaged metrics (for multi-class)

* Displays:

  * Confusion matrix (as heatmap)
  * For binary scenario with `predict_proba`:

    * ROC curve with AUC
    * Precision–Recall curve and PR-AUC

This function is reused across:

* Closed-world scenarios (95 classes).
* Open-world binary detection (monitored vs unmonitored).
* Open-world multi-class + reject (95 + 1 classes).
* Two-stage hierarchical modeling.

---

## 7. Scenario-Based Experiments (final_modeling_code.ipynb)

The notebook `final_modeling_code.ipynb` uses the cleaned, feature-selected datasets
(e.g., `X_mon_clean`, `y_mon_clean`, `X_unmon_clean`, `X_all`, `y_multi`)
to run several **scenarios**.

### 7.1 Scenario 1 – Closed-World Model Comparison & RF Tuning

* Split monitored data (`X_mon_clean`, `y_mon_clean`) into train/test.

* Apply `StandardScaler`.

* Compare multiple classifiers:

  * `DecisionTreeClassifier`
  * `KNeighborsClassifier`
  * `GaussianNB`
  * `LogisticRegression`
  * `SVC (RBF)`
  * `RandomForestClassifier`
  * `MLPClassifier` (simple neural net)

* For each model:

  * Train on scaled training data
  * Evaluate accuracy, macro F1, and runtime
  * Summarize in a leaderboard DataFrame + bar plot

* Perform **RandomizedSearchCV** on Random Forest:

  * Search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `class_weight`
  * Optimize for accuracy on the closed-world task
  * Evaluate tuned RF on the held-out test set:

    * Accuracy
    * Macro F1
    * Confusion matrix
    * (Optional) classification report

### 7.2 Scenario 2 – Open-World Binary Detection (Monitored vs Unmonitored)

* Construct a binary dataset:

  * Monitored → label `1`
  * Unmonitored → label `-1`

* Use a common train/test split and `StandardScaler`.

* Compare multiple **class-imbalance strategies**:

  1. **Undersampling**:

     * Downsample the majority class to match the minority.
  2. **SMOTE + Tomek** (or plain SMOTE as fallback).
  3. **Class-weighted RandomForest**.
  4. **Borderline-SMOTE + EditedNearestNeighbours**.
  5. **ADASYN + RF**.
  6. **BalancedRandomForestClassifier**.
  7. **XGBoost** with `scale_pos_weight`.

* For each strategy:

  * Train the classifier on the resampled / reweighted data.
  * Evaluate using a helper `eval_model(name, y_test, y_pred, probas)`:

    * Full classification report
    * **Unmonitored F1-score** (label = `-1`)
    * ROC-AUC when probabilities are available.

* Additionally, for the SMOTE+Tomek RF model:

  * Use `precision_recall_curve` on the **unmonitored class** (`-1`).
  * Compute F1 for each threshold and select the **best threshold**.
  * Apply this threshold to convert probabilities to final labels.
  * Recompute classification report and confusion matrix.

This scenario focuses on **robust detection of unmonitored traffic** (rejecting unknown sites).

### 7.3 Scenario 3 – Open-World Multi-Class (95 Monitored + Reject Class)

* Define labels:

  * Monitored sites: `0–94`
  * Unmonitored traffic: `-1` (reject)

* Construct `X_all` and `y_multi = [0–94, -1]`.

* Stratified train/test split.

* Apply `StandardScaler`.

* Tune a class-weighted `RandomForestClassifier` using `RandomizedSearchCV`:

  * Search over:

    * `n_estimators`
    * `max_depth`
    * `min_samples_split`
    * `max_features`

  * Optimize **macro F1-score** to account for class imbalance.

* Evaluate the tuned RF with `evaluate_model_performance(..., scenario="multi")`:

  * Accuracy
  * Macro F1
  * Unmonitored recall (how well `-1` is recovered)
  * Confusion matrix spanning 96 labels (`-1` + 0–94)

This scenario treats the open world as a **multi-class + reject** problem.

### 7.4 Scenario 4 – Two-Stage Hierarchical Open-World Model

A more realistic architecture is implemented as a **two-stage pipeline**:

1. **Stage 1 – Binary Detector (Monitored vs Unmonitored)**

   * Build binary labels from `y_train`:

     * Unmonitored → `-1`
     * Monitored → `1`

   * Apply **SMOTE+Tomek** (or SMOTE fallback) on the training data.

   * Train a binary `RandomForestClassifier` with:

     * `n_estimators=500`, `max_depth=None`
     * `class_weight='balanced'`

2. **Stage 2 – Closed-World Identifier (95-class)**

   * Filter training samples with `y_train != -1`.
   * Train a 95-class Random Forest to classify among monitored sites `0–94`.

3. **Two-Stage Inference**

   * On the test set:

     * Stage 1 predicts whether a sample is monitored or unmonitored.
     * If predicted as monitored (`1`), pass to Stage 2 for 95-class identification.
     * Otherwise, label remains `-1`.

4. **Final Evaluation**

   * Overall accuracy
   * Macro F1
   * Unmonitored recall (reject rate)
   * Unmonitored-focused confusion analysis:

     * True unmonitored: `-1 → -1`
     * False alarms: `-1 → monitored-class`
     * False alarm rate

This hierarchical design mirrors **real-world deployment**, where we first detect whether traffic belongs to any monitored target set, and only then identify the specific site.

---

## 8. How to Run (Detailed)

### 8.1 Environment

```bash
python >= 3.8
pip install -r requirements.txt
```

Key dependencies:

* `numpy`, `pandas`
* `matplotlib`, `seaborn`
* `scikit-learn`
* `scipy`
* `imbalanced-learn`
* `xgboost` (for some imbalance strategies)

### 8.2 Dataset Setup

Place the dataset under `data/` (local run) or mount Google Drive (Colab) as described in **0.2 Dataset Setup**.

Example (local):

```python
MON_PATH = "data/mon_standard.pkl"
UNMON_PATH = "data/unmon_standard10.pkl"
```

### 8.3 Baseline (Closed-world, 10 Features)

```bash
jupyter notebook baseline_feature10.ipynb
```

Run all cells to:

* Load monitored data from `MON_PATH`.
* Extract 10 manually engineered statistical features.
* Train a closed-world Random Forest baseline.
* Compute feature importance and simple ablation.
* Optionally, perform basic IQR-based cleaning.

### 8.4 Feature Selection Pipeline (37 → 27 → 26 Features)

```bash
jupyter notebook final_code_featureSelection.ipynb
```

Run all cells to:

* Load monitored data.
* Extract 37 advanced traffic features.
* Build correlation matrices & RF importances.
* Perform correlation-based pruning (for analysis), zero-importance pruning (37 → 27).
* Apply IQR-based outlier removal.
* Run ablation on the 27-feature set and small combination search (27 → 26).
* Fairly compare **37 / 27 / 26** models on a common cleaned subset.

The resulting 27- and 26-feature sets are used as **candidate feature sets** for final modeling.

### 8.5 Final Modeling & Scenario Experiments

```bash
jupyter notebook final_modeling_code.ipynb
```

Run all cells to:

* Compare multiple classifiers on the closed-world task.
* Tune Random Forest hyperparameters with `RandomizedSearchCV`.
* Conduct open-world binary detection experiments with multiple imbalance strategies.
* Perform threshold tuning based on the unmonitored class F1.
* Run open-world multi-class experiments (95 monitored + reject).
* Train and evaluate a hierarchical two-stage model (detection → identification).

> **Note:** Before running this notebook, please upload the provided
`processed_traffic_data_26feats.pkl` to the working directory.
No raw dataset or feature-selection preprocessing is required.
---
