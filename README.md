# Closed- & Open-World Website Fingerprinting
**Team 07 Molock**

> 🔎 Machine Learning Team Project — **Closed-world baseline**, plus a **37 → 27 → 26 feature selection pipeline** for Closed- & Open-World evaluation.

This repository contains:

- A **closed-world Random Forest baseline** with simple handcrafted features.
- A **final model** that:
  - Extracts **37 statistical / burst-based features** from packet sequences.
  - Reduces them to **27** and then **26** features using:
    - Correlation-based pruning  
    - Zero-importance pruning  
    - IQR-based outlier removal  
    - Ablation study and small combinatorial search
  - Prepares models that can be evaluated in **closed-world** and **open-world** scenarios.


---

## 1. Repository Structure

```bash
.
├── baseline_feature10.ipynb      # Simple closed-world baseline (10 handcrafted features)
├── final_model_feature37.ipynb   # Final 37→27→26 feature selection + RF comparison
├── data/
│   ├── mon_standard.pkl          # place monitored dataset here
│   ├── unmon_standard10.pkl      # place unmonitored dataset 
└── README.md
```
<br/>
Note: Due to size and privacy constraints, the dataset files (`mon_standard.pkl`, `unmon_standard10.pkl`) are not included in this repository. Please place them under the data/ directory as shown above.

---

## 2. Data Description

### 2.1 Monitored Data (`mon_standard.pkl`)

* Preprocessed monitored traffic traces.
* Each `data[i]` contains multiple samples (traces) for the *i-th* URL.
* Each sample is a sequence of integers `c`, where:

  * **Sign** encodes packet **direction**:

    * `c > 0` → outgoing
    * `c < 0` → incoming
  * **Absolute value** is used as a **timestamp** in the current code.

The loader returns:

* `X1`: list of timestamp sequences (`t_seq`)
* `X2`: list of direction × fixed-size sequences (`s_seq`, values ±512)
* `y`: class labels

Labeling options:

```python
USE_SUBLABEL = False   # If True: each URL is a separate class; if False: site-level label
URL_PER_SITE = 10
TOTAL_MON_URLS = 950   # Example: 95 sites × 10 URLs
```

* If `USE_SUBLABEL = False`:
  `label = i // URL_PER_SITE` (site-level)
* If `USE_SUBLABEL = True`:
  `label = i` (URL-level)

### 2.2 Unmonitored Data (`unmon_standard10.pkl`)

* Contains unmonitored / background traffic.
* Used for **open-world** experiments in the final model.
* The current README only prepares the feature-selection side; open-world scenario experiments are left as placeholders.

---

## 3. Closed-World Baseline (10 Features)

The baseline (in `baseline_feature10.ipynb` or equivalent) uses a **simpler feature extractor** to construct **10 features** per trace, such as:

* Duration
* Inter-arrival time statistics
* Cumulative bytes
* Burst count & average burst size
* In/out packet counts & ratios
* Std of outgoing inter-arrival times

Then it:

1. Trains a **RandomForestClassifier** with all 10 features.
2. Computes:

   * Correlation matrix and identifies strongly correlated pairs.
   * Feature importances from RF.
3. Runs a **leave-one-out ablation study**:

   * Remove each feature in turn and measure accuracy change.
4. Selects a smaller, high-impact subset of features.
5. Applies **IQR-based outlier removal** and compares accuracy before / after.
6. Performs **hyperparameter tuning** with `HalvingRandomSearchCV`.

This closed-world baseline acts as a **simple, feature-interpretable reference model.**

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

From `ipi = diff(X1)`:

* `IPI_Mean`, `IPI_Std`, `IPI_Max`, `IPI_Min`
* `IPI_Skew`, `IPI_Kurt`
* `IPI_P10`, `IPI_P25`, `IPI_P50`, `IPI_P75`, `IPI_P90`

#### 4.3 Packet size distribution features

From `abs(X2)`:

* `Size_P10`, `Size_P25`, `Size_P50`, `Size_P75`, `Size_P90`

#### 4.4 Burst-level features

Bursts are defined as **maximal runs of packets with the same direction**.

For burst length (`bl`) and burst volume (`bv`):

* `BurstLen_Count` — number of bursts
* `BurstLen_Mean`, `BurstLen_Std`, `BurstLen_Max`, `BurstLen_Min`
* `BurstVol_Mean`, `BurstVol_Std`, `BurstVol_Max`, `BurstVol_Min`

#### 4.5 Cumulative volume features

From `cumsum(abs(X2))`:

* `CumSum_Mean`, `CumSum_Std`

Overall:

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

This section describes the **full feature selection pipeline** implemented in the final model.

### 5.1 Step 1 – Correlation Analysis & Base RF

1. Compute the **correlation matrix** of the 37 features.
2. Identify **highly correlated feature pairs** with `|r| ≥ 0.9` and store them in `high_corr_pairs`.
3. Train a baseline `RandomForestClassifier` on all 37 features:

   * Split the dataset once (`train_test_split` with `stratify=y`).
   * Fit RF (e.g., `n_estimators=200`).
   * Compute:

     * Baseline accuracy
     * Feature importances (`importances: Series[feature → importance]`)
4. Plot:

   * Correlation heatmap
   * Horizontal bar chart of feature importances

This gives a **first view of redundancy** (correlated features) and a **ranking of importance**.

---

### 5.2 Step 2 – Correlation-Based Pruning

Goal: remove redundant features that are **highly correlated** and **less important**.

Procedure:

1. Use a fixed train/test split (indices) for fairness.
2. Compute baseline accuracy with all 37 features (`base_acc_corr`).
3. For each pair `(a, b, r)` in `high_corr_pairs`:

   * Compare `importances[a]` and `importances[b]`.
   * Add the **less important** feature to `to_drop_corr`.
4. Define:

   * `kept_cols = all_features - to_drop_corr`
   * `X_pruned_corr = X_df[kept_cols]`
5. Train RF again on `X_pruned_corr` with the same indices:

   * Evaluate `pruned_acc`.
   * Compare `Δ = pruned_acc - base_acc_corr`.

This step checks **how much we can remove purely based on correlation**, while tracking accuracy.<br/>

Although correlation-based pruning successfully reduced the number of features, it resulted in a performance drop compared to the full 37-feature baseline. Since our goal was to reduce dimensionality without sacrificing accuracy, this approach was not adopted in the final feature set.

---

### 5.3 Step 3 – Zero-Importance Pruning (37 → 27)

Next, features with **near-zero RF importance** are removed.

1. Identify zero-importance features:

   ```python
   zero_importance_feats = importances[importances <= 1e-6].index.tolist()
   ```

2. Measure baseline accuracy with **all 37 features** on the same split.

3. Drop `zero_importance_feats`:

   ```python
   X_pruned = X_df.drop(columns=zero_importance_feats)   # → 27 features
   ```

4. Re-train RF and compute accuracy for the 27-feature set.

5. Compare:

   * Accuracy before vs after pruning
   * Number of features (`37 → 27`)

This yields a **27-feature pruned set**, where features with **essentially no contribution** (in importance) are removed.<br/>

After removing the near-zero importance features, the model trained on the 27-feature set showed slightly improved accuracy compared to the full 37-feature baseline. Therefore, we decided to retain the 27-feature configuration as the primary pruned feature set.

---

### 5.4 Step 4 – IQR Outlier Removal & Accuracy Comparison

Goal: compare different combinations of **feature sets** (full vs pruned) and **outlier handling** (raw vs cleaned).

#### 5.4.1 IQR-based outlier cleaning

For a given DataFrame `df`:

* For each feature:

  * Compute `Q1`, `Q3`, `IQR = Q3 − Q1`.
  * Set bounds `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
  * Keep samples within the bounds.
* Combine masks across all features (logical AND).
* Return cleaned DataFrame and mask.

Applied to:

* `X_full` — all 37 features
* `X_pruned` — the 27-feature pruned set

We obtain:

* `X_full_clean`, `y_full_clean`
* `X_pruned_clean`, `y_pruned_clean`

#### 5.4.2 RF accuracy for four configurations

For each of the four combinations:

1. `Full 37 / Raw`
2. `Full 37 / Clean`
3. `Pruned 27 / Raw`
4. `Pruned 27 / Clean`

We:

* Perform a stratified train/test split.
* Train an RF with fixed hyperparameters.
* Record the accuracy.

Results are summarized in a small table (`acc_summary`), including:

* `config`
* `n_features`
* `n_samples`
* `accuracy`

The best configuration (`best_config`) is selected based on maximum accuracy.

For subsequent visualization:

* `best_raw_df` and `best_clean_df` are chosen to match the best configuration.
* A large grid of **Raw vs Clean boxplots** (log-transformed) is drawn for all features in the selected set to show distribution tightening after outlier removal.

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
4. Compile an **ablation summary table** and a **bar plot of Δ accuracy**.

Interpretation:

* Features whose removal **significantly decreases** accuracy are **important**.
* Features whose removal **does not hurt** (or slightly improves) accuracy are good candidates for further pruning.

---

### 5.6 Step 6 – Searching for a Compact 26-Feature Set

Based on the ablation results, a small subset of candidate features to drop is chosen:

```python
to_drop = ["IPI_Mean", "BurstLen_Max", "BurstVol_Mean", "BurstLen_Std"]
```

Then:

1. For all non-empty subsets `combo` of `to_drop`:

   * Drop `combo` from `X_pruned_clean`.
   * Retrain RF on the same split.
   * Record:

     * `drop_list`
     * `num_feats`
     * `accuracy`
2. Sort results by accuracy.
3. Inspect the best combinations to find a **good trade-off** between:

   * Fewer features
   * Minimal or improved accuracy

In the final decision, the combination that resulted in 26 features—by removing one additional feature such as IPI_Mean—produced the highest evaluation score across the tested subsets. Therefore, this 26-feature configuration was included as a candidate for further comparison.

---

### 5.7 Step 7 – Fair Comparison: 37 vs 27 vs 26 (Common Clean Mask)

To compare the 37-, 27-, and 26-feature models fairly:

1. Start from the original 37-feature DataFrame `X_37`.

2. Apply **a single, common IQR cleaning pass** to `X_37`:

   * Get `X_37_clean` and `common_mask`.
   * Apply the same `common_mask` to `y` → `y_clean`.

3. Define:

   ```python
   zero_importance_feats = [
       "IPI_Min", "IPI_P10", "IPI_P25",
       "Size_P90", "Size_P50", "Size_P75", "Size_P10", "Size_P25",
       "BurstLen_Min", "BurstVol_Min"
   ]

   X_27_clean = X_37_clean.drop(columns=zero_importance_feats)  # 27 features
   X_26_clean = X_27_clean.drop(columns=["IPI_Mean"])           # 26 features
   ```

4. Using the **same clean dataset and the same train/test split**:

   * Train RF with 37 features.
   * Train RF with 27 features.
   * Train RF with 26 features.

5. Evaluate each model with a unified helper:

   ```python
   evaluate_model_performance(
       model, X_test, y_test,
       model_name="RF - XX feats",
       scenario="multi"
   )
   ```

   * `scenario="multi"` uses macro-averaged Precision, Recall, F1.
   * Optionally logs open-world related metrics if class `-1` exists in labels.

This yields a **direct, fair comparison** between the full and reduced feature sets, confirming that moving to **27 or 26 features** does not significantly harm (and may even slightly improve) performance, while reducing dimensionality.

---

## 6. Evaluation Utilities

The final model includes a general-purpose evaluation function:

```python
evaluate_model_performance(model, X_test, y_test, model_name, scenario="binary" or "multi")
```

Features:

* Computes:

  * Accuracy
  * Precision, Recall, F1 (binary or macro)
  * Optional ROC-AUC, PR-AUC (for binary)
  * Unmonitored recall in open-world settings (if label `-1` is used)
* Prints:

  * A small metrics table (as a DataFrame)
  * Classification report (for binary)
* Plots:

  * Confusion matrix
  * ROC & Precision–Recall curves (for binary)

This utility is reused for:

* **Closed-world, multi-class evaluation** (`scenario="multi"`)
* **Open-world, binary monitored vs unmonitored detection** (`scenario="binary"`)

---

## 7. Scenario-Based Experiments (TBD)

>
임시내용 작성
### 7.1 Closed-World Scenario


### 7.2 Open-World Scenario


### 7.3 Ablation & Sensitivity Analyses


---

## 8. How to Run

### 8.1 Environment

```bash
python >= 3.8

pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

### 8.2 Baseline (Closed-world, 10 Features)

```bash
jupyter notebook baseline_feature10.ipynb
```

Run all cells to:

* Load monitored data
* Extract 10 features
* Train RF and perform basic feature analysis and IQR cleaning

Or, if converted to a script:

```bash
python baseline_feature10.py
```

### 8.3 Final Model (37 → 27 → 26 Features)

```bash
jupyter notebook final_model_feature37.ipynb
```

Run all cells to:

* Load monitored data
* Extract 37 features
* Perform correlation & importance analysis
* Apply zero-importance & correlation-based pruning
* Apply IQR outlier removal
* Run ablation & combination search
* Compare 37 / 27 / 26 feature models on a common clean dataset

Scenario-based experiments using `UNMON_PATH` can be added to a separate notebook or appended to the same one under **Section 7**.

### 8-4. Run in Google Colab

You can also open the notebook in Colab using the buttons below.

baseline_feature10.ipynb : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyungh2e2e/Molock_ML25-2/blob/main/baseline_feature10.ipynb)

---





