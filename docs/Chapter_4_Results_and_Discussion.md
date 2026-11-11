# CHAPTER IV: RESULTS AND DISCUSSION

**Study Title:** *Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis*  
**Course:** BSIT 3A – IT 119: Data Scalability and Analytics  
**Researchers:** Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano  
**Institution:** Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines  

---

## 4.1 Multi-Source Data Integration, Cleansing, and Attrition Audit

The primary CNSC Google Forms survey ($123$ raw responses) was harmonized with three published public datasets on blended and online learning in higher education, giving **$1,530$ raw records** and **$1,427$ cleaned records** ($93.27\%$ retention; $103$ invalid/screened records removed) for PySpark processing. Table 4.1 summarizes the exact per-study screening audit (`study_rows_lost_and_reasons.csv`).

**Table 4.1**  
*Data Integration, Screening, and Retention Log across Unified Data Sources ($N = 1,427$)*

| Source Study Identifier | Study Role & Citation | Raw ($n$) | Cleaned ($n$) | Lost ($n$) | Retention (%) | Screening & Attrition Reasons |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Main_Study_PH_CNSC_Blended_221`** | Main Study (PH CNSC, Chavez et al., 2026) | 123 | 117 | 6 | 95.12% | Straight-lining across all 11 Likert items with blank open-ended text ($n=5$); Duplicate submission ($n=1$). |
| **`Public_S2_Le_2026_VN_Blended`** | Public Study (Le et al., 2026, Vietnam) | 580 | 508 | 72 | 87.59% | Straight-lining / zero-variance responses screened out ($n=72$). |
| **`Public_S3_Irwanto_2024_ID_SRL`** | Public Study (Irwanto, 2024, Indonesia) | 577 | 573 | 4 | 99.31% | Straight-lining / zero-variance responses screened out ($n=4$). |
| **`Public_S4_Makassar_2021_ID_Online`** | Public Study (Univ. Negeri Makassar, 2021) | 250 | 229 | 21 | 91.60% | Straight-lining / zero-variance responses screened out ($n=21$). |
| **TOTAL UNIFIED MASTER DATASET** | **All Combined (Main + 3 Public)** | **1,530** | **1,427** | **103** | **93.27%** | **Strict Stage 2 screening audit logged in `stage2_cleaning_loss_log.csv`.** |

[insert fig1_data_cleaning_summary]  
*(Insert Figure 1: `output/data-cleaning/analysis/figures/fig1_data_cleaning_summary.png` — Multi-Source Data Ingestion & Stage 2 Screening Loss)*

---

## 4.2 Construct Comparison Across Integrated Studies

Table 4.2 compares the composite construct means across the four integrated studies (`integration_summary_by_source.csv`). Across every study, the **Power/Fuel Crisis Mitigation Index (`crisis_index`) exceeds the Academic Learning Index (`learning_index`)**, producing a consistent positive Crisis–Learning Gap ($+1.101$ overall).

**Table 4.2**  
*Composite Construct Means by Source Study ($N = 1,427$)*

| Source Study | Clean ($n$) | Mean Learning Index | Mean Adaptability Index | Mean Crisis Index | Mean Gap ($\Delta$) | Mean Support (`support_continue`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Main Study (PH CNSC)** | 117 | 3.322 | 3.598 | 3.949 | +0.627 | 3.419 |
| **Le et al. (2026, Vietnam)** | 508 | 3.868 | 3.834 | 5.000 | +1.132 | 4.868 |
| **Irwanto (2024, Indonesia)** | 573 | 3.883 | 3.844 | 5.000 | +1.117 | 4.883 |
| **Makassar (2021, Indonesia)** | 229 | 3.760 | 3.893 | 4.981 | +1.221 | 4.603 |
| **Overall Unified Corpus** | **1,427** | **3.812** | **3.828** | **4.914** | **+1.101** | **4.713** |

[insert fig2_unified_indices_by_study]  
*(Insert Figure 2: `output/data-cleaning/analysis/figures/fig2_unified_indices_by_study.png` — Composite Construct Scores across Integrated Datasets)*

---

## 4.3 Item-Level Likert Distributions and Composite Construct Indices

Table 4.3 presents the descriptive statistics and agreement percentages across canonical items and derived composite indices (`analysis1_descriptive_stats.csv` and `analysis2_likert_distribution.csv`).

**Table 4.3**  
*Descriptive Statistics and Likert Agreement Distributions ($N = 1,427$)*

| Variable / Item | Label | Mean ($\bar{x}$) | $SD$ | Median |
| :--- | :--- | :---: | :---: | :---: |
| `reduce_electricity` | Reduced electricity use at school | **4.928** | 0.354 | 5.0 |
| `reduce_fuel` | Reduced household fuel expenses | **4.921** | 0.380 | 5.0 |
| `crisis_effectiveness` | Effective vs. power/fuel crisis | **4.894** | 0.443 | 5.0 |
| `support_continue` | Support continuing 2-2-1 as is | **4.713** | 0.776 | 5.0 |
| `f2f_focus` | Focus is stronger on F2F days | **4.448** | 0.942 | 5.0 |
| `transition_comfort` | Comfortable switching F2F/online/async | **3.832** | 0.767 | 4.0 |
| `schedule_fit` | Fits daily schedule/responsibilities | **3.824** | 0.749 | 4.0 |
| `stay_on_track` | Can complete requirements, stay on track | **3.816** | 0.742 | 4.0 |
| `understanding` | Improved understanding of lessons | **3.812** | 0.748 | 4.0 |
| `academic_performance` | Academic performance vs. full F2F | **3.809** | 0.751 | 4.0 |
| **`crisis_index`** | **Power/Fuel Crisis Mitigation Index** | **4.914** | **0.342** | **5.000** |
| **`adaptability_index`** | **Modality Adaptability Index** | **3.828** | **0.677** | **4.000** |
| **`learning_index`** | **Academic Learning Index** | **3.812** | **0.686** | **3.667** |
| **`crisis_learning_gap`** | **Crisis–Learning Gap ($\text{crisis} - \text{learning}$)** | **+1.101** | **0.697** | **+1.333** |

[insert fig3_likert_diverging]  
*(Insert Figure 3: `output/data-cleaning/analysis/figures/fig3_likert_diverging.png` — Likert Distributions for Learning vs. Crisis Mitigation)*

---

## 4.4 The Crisis–Learning Gap and Subgroup Stratification

Students with *Very reliable* home internet achieve significantly higher learning index scores ($M = 3.966$) and report fewer challenges than peers with *Unreliable home internet* ($M = 3.193$). Commuting students living with family report strong household fuel savings (`reduce_fuel` $= 4.921$) under the 2-2-1 schedule, while boarding and dormitory students emphasize campus Wi-Fi and power stability needs.

[insert fig4_index_by_internet_and_residence]  
*(Insert Figure 4: `output/data-cleaning/analysis/figures/fig4_index_by_internet_and_residence.png` — Scores by Home Internet Reliability and Student Residence)*

---

## 4.5 Modality Challenges, Improvement Priorities, and Correlations

**Bivariate Pearson Correlations (`analysis5_correlations.csv`)**:
- `crisis_index` vs. `support_continue`: **$r = +0.6164$** (strongest predictor of continuation support)
- `learning_index` vs. `support_continue`: **$r = +0.2666$**
- `adaptability_index` vs. `support_continue`: **$r = +0.2228$**
- `schedule_fit` vs. `transition_comfort`: **$r = +0.6121$**
- `reduce_fuel` vs. `crisis_effectiveness`: **$r = +0.8145$**

[insert fig5_open_ended_themes]  
*(Insert Figure 5: `output/data-cleaning/analysis/figures/fig5_open_ended_themes.png` — Reported Challenges & Institutional Improvement Priorities)*

[insert fig6_correlation_and_gap]  
*(Insert Figure 6: `output/data-cleaning/analysis/figures/fig6_correlation_and_gap.png` — Correlation Matrix & Crisis-Learning Gap Typology)*

---

## 4.6 Unsupervised Machine Learning: Student Personas ($K$-Means & Mean-Shift)

### 4.6.1 $K$-Means Clustering ($k = 4$ Student Personas)
Evaluating $k \in [2, 6]$ on standardized construct features (`kmeans_evaluation_k2_k6.csv`) revealed optimal persona segmentation at **$k = 4$ (Silhouette $= 0.3780$)**. Table 4.8 profiles the four student personas discovered at $k = 4$ (`kmeans_profiles.csv`).

**Table 4.8**  
*K-Means Student Behavioral Personas ($k = 4, N = 1,427$)*

| Cluster ID | Persona Name | Count ($n$) | Share (%) | Mean Learning | Mean Adaptability | Mean Crisis | Mean F2F Focus | Mean Support |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Adapted Blended Supporters** | 395 | 27.68% | 3.910 | 3.949 | 4.995 | 4.977 | **4.987** |
| **1** | **Balanced Hybrid Adapters** | 601 | 42.12% | **4.131** | **4.196** | 4.991 | 4.737 | 4.672 |
| **2** | **Crisis-Pragmatic Neutrals** | 356 | 24.95% | 3.348 | 3.219 | 4.994 | 3.458 | 4.865 |
| **3** | **Connectivity-Constrained Critics** | 75 | 5.26% | **2.951** | **3.133** | 3.489 | 4.040 | **2.880** |

[insert fig1_kmeans_elbow_silhouette]  
*(Insert Figure 7: `output/unsupervised-ml-algo/figures/fig1_kmeans_elbow_silhouette.png` — K-Means Elbow & Silhouette Optimization)*

[insert fig2_kmeans_pca_clusters]  
*(Insert Figure 8: `output/unsupervised-ml-algo/figures/fig2_kmeans_pca_clusters.png` — K-Means Student Personas in 2D PCA Space)*

[insert fig3_kmeans_cluster_profiles]  
*(Insert Figure 9: `output/unsupervised-ml-algo/figures/fig3_kmeans_cluster_profiles.png` — Construct Profiles Across K-Means Personas)*

### 4.6.2 Mean-Shift Density Clustering
Non-parametric Mean-Shift clustering (`bandwidth = 1.8475`, Silhouette $= 0.2931$) automatically discovered **45 natural density modes** (`mean_shift_profiles.csv`), with the primary mode ($59.15\%, n=844$) and secondary mode ($27.26\%, n=389$) demonstrating high continuation support and learning effectiveness, while smaller outlier modes isolate connectivity-constrained respondents.

[insert fig4_mean_shift_clusters]  
*(Insert Figure 10: `output/unsupervised-ml-algo/figures/fig4_mean_shift_clusters.png` — Mean-Shift Density Modes in 2D PCA Space)*

---

## 4.7 Supervised Machine Learning: Regression & Classification Models

Table 4.9 summarizes the performance of all six supervised machine learning algorithms (`supervised_models_comparison.csv`).

**Table 4.9**  
*Supervised Machine Learning Model Comparison ($N = 1,427$; 75% Train / 25% Test Split)*

| Algorithm | Task Type | Primary Metric | Secondary Metric / Configuration |
| :--- | :--- | :---: | :--- |
| **Simple Linear Regression (SLR)** | Regression (`support_continue`) | **$R^2 = 0.0446$** | $\text{RMSE} = 0.7585, \text{MAE} = 0.4840$ ($\hat{y} = 0.3048x + 3.5688$) |
| **Multiple Linear Regression (MLR)** | Regression (`support_continue`) | **$R^2 = 0.4561$** | $\text{RMSE} = 0.5723, \text{MAE} = 0.3373$ (6 Domain Predictors) |
| **K-Nearest Neighbors (KNN)** | Classification (`support_category`) | **Accuracy = 94.40%** ($k=5$) | **Optimal $k = 12$ achieves 95.52% accuracy** |
| **Support Vector Machine (SVM)** | Classification (`support_category`) | **Accuracy = 94.12%** | RBF Kernel, $C = 1.0$, scaled features |
| **Decision Tree Classifier** | Classification (`support_category`) | **Accuracy = 93.84%** | `max_depth = 3`, Gini criterion |
| **Gaussian Naive Bayes** | Classification (`support_category`) | **Accuracy = 92.72%** | Empirical Gaussian priors |

### 4.7.1 Regression Insights (SLR vs. MLR)
- **Simple Linear Regression (`slr_metrics.csv`)**: Evaluates academic learning as a baseline predictor of support for continuing the 2-2-1 modality ($\beta_1 = +0.3048, R^2 = 0.0446$).
- **Multiple Linear Regression (`mlr_coefficients.csv`)**: Combining Academic Learning, Modality Adaptability, Crisis Mitigation, Internet Reliability, and Modality Challenges increases explained variance to **$45.61\%$** ($R^2 = 0.4561, \text{RMSE} = 0.5723$). The dominant positive predictor is the **Power/Fuel Crisis Mitigation Index ($\beta = +0.8259$)**, followed by **Internet Reliability ($\beta = +0.3864$)**, **Academic Learning Index ($\beta = +0.1183$)**, and **Modality Adaptability ($\beta = +0.1167$)**.

[insert slr_fit]  
*(Insert Figure 12: `output/supervised-ml-algo/figures/slr_fit.png` — Simple Linear Regression Fit)*

[insert mlr_fit]  
*(Insert Figure 13: `output/supervised-ml-algo/figures/mlr_fit.png` — Multiple Linear Regression Actual vs. Predicted)*

### 4.7.2 Classification Insights (Decision Tree, KNN, Naive Bayes, SVM)
- **Decision Tree (`feature_importances.csv`)**: Achieves **$93.84\%$** accuracy (`max_depth=3`), where `crisis_index` represents **$66.04\%$** of split importance, `adaptability_index` represents **$17.85\%$**, and `learning_index` represents **$13.05\%$**.
- **K-Nearest Neighbors (`knn_metrics.csv`)**: Achieves **$94.40\%$** accuracy at baseline $k = 5$ and peaks at **$95.52\%$** at optimal $k = 12$.
- **Gaussian Naive Bayes (`naive_bayes_metrics.csv`)**: Achieves **$92.72\%$** accuracy with high recall across cohorts.
- **Support Vector Machine (`svm_metrics.csv`)**: Achieves **$94.12\%$** test accuracy using an RBF kernel in scaled feature space.

[insert decision_tree]  
*(Insert Figure 14: `output/supervised-ml-algo/figures/decision_tree.png` — Decision Tree Classifier)*

[insert knn_accuracy_k]  
*(Insert Figure 15: `output/supervised-ml-algo/figures/knn_accuracy_k.png` — KNN Accuracy Across k = 1..15)*

[insert naive_bayes_cm]  
*(Insert Figure 16: `output/supervised-ml-algo/figures/naive_bayes_cm.png` — Gaussian Naive Bayes Confusion Matrix)*

[insert svm_decision_boundaries]  
*(Insert Figure 17: `output/supervised-ml-algo/figures/svm_decision_boundaries.png` — SVM Decision Boundaries in Latent PCA Space)*
