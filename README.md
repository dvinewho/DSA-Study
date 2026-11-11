# Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis

**Course:** BSIT 3A – IT 119: Data Scalability and Analytics  
**Authors:** Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano  
**Institution:** Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines  

This project combines our primary Google Forms survey on CNSC's 2-2-1 blended learning schedule with three published, peer-reviewed public datasets on blended/hybrid learning. It executes a comprehensive 5-stage PySpark data cleaning & integration pipeline, 6 supervised machine learning algorithms, and 2 unsupervised clustering models.

---

## Multi-Source Datasets (1,530 Raw &rarr; 1,427 Cleaned Records)

All datasets are real, verified datasets downloaded from Mendeley Data and open survey responses:

| # | Dataset Source | Repository / Reference | Raw Records | Screened Rows | Cleaned Records | Status |
|---|----------------|------------------------|------------:|--------------:|----------------:|--------|
| 1 | Primary CNSC Survey (`dataset/Perceptions of College Students ... .csv`) | Our Google Form (CNSC BSIT 3A) | 123 | 6 | 117 | Real survey data |
| 2 | Le et al. (2026), Vietnam Blended Learning Perceptions | Mendeley Data [`10.17632/tdsspksw83.1`](https://doi.org/10.17632/tdsspksw83.1) | 580 | 72 | 508 | Real published data |
| 3 | Irwanto (2024), Indonesia SRL in Blended Learning | Mendeley Data [`10.17632/7bsvck73ng.1`](https://doi.org/10.17632/7bsvck73ng.1) | 577 | 4 | 573 | Real published data |
| 4 | Makassar (2021), Student Perception About Online Learning | Mendeley Data [`10.17632/8myyf5t45z.3`](https://doi.org/10.17632/8myyf5t45z.3) | 250 | 21 | 229 | Real published data |
| | **Unified Master Architecture Total** | | **1,530** | **103** | **1,427** | **100% Real Data** |

### Data Screening Criteria
- **CNSC Primary Survey:** 1 duplicate submission (same email) and 5 straight-line responses (identical Likert responses across all items with blank qualitative feedback) screened out.
- **Le et al. (2026):** 72 straight-line / zero-variance responses screened out.
- **Irwanto (2024):** 4 straight-line / zero-variance responses screened out.
- **Makassar (2021):** 21 straight-line / zero-variance responses screened out.
- Detailed citations, DOIs, and variable mappings are documented in [`dataset/public/citation.txt`](dataset/public/citation.txt).

---

## Project Structure

```
DSA/
├── AGENTS.md                      # Run-order guide and project specification
├── README.md                      # Project documentation and setup instructions
├── dataset/
│   ├── Perceptions of College Students ... Power Fuel Crisis.csv   # Primary survey (123 rows)
│   └── public/                    # 3 real published datasets + citation.txt
│       ├── Le_2026_VN_Blended_Learning_Perceptions/
│       ├── Irwanto_2024_ID_SRL_Blended_Learning/
│       ├── Makassar_2021_Student_Perception_Online_Learning/
│       └── citation.txt
├── code/
│   ├── data-cleaning/             # 5-stage PySpark pipeline
│   │   ├── pipeline_utils.py
│   │   ├── 01_data_integration.py
│   │   ├── 02_data_cleansing.py
│   │   ├── 03_data_transformation.py
│   │   ├── 04_data_reduction.py
│   │   └── 05_data_analysis_and_visualization.py
│   ├── supervised-ml-algo/        # SLR, MLR, Decision Tree, KNN, Naive Bayes, SVM
│   │   ├── main.py
│   │   ├── simple_linear_regression.py
│   │   ├── multiple_linear_regression.py
│   │   ├── decision_tree.py
│   │   ├── knn.py
│   │   ├── naive_bayes.py
│   │   └── svm.py
│   └── unsupervised-ml-algo/      # K-Means (k=4), Mean-Shift
│       ├── main.py
│       ├── kmeans.py
│       └── mean_shift.py
├── output/                        # Generated CSV tables, figures, key insights
│   ├── data-cleaning/
│   │   ├── integration/
│   │   ├── cleansing/
│   │   ├── transformation/
│   │   ├── reduction/
│   │   └── analysis/figures/
│   ├── supervised-ml-algo/figures/
│   └── unsupervised-ml-algo/figures/
└── docs/                          # Chapters 1–4 and final paper draft
```

---

## Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Java OpenJDK 17 or 21 (PySpark requirement). If `JAVA_HOME` is unset, `pipeline_utils.py` automatically detects local OpenJDK installations in `~/.jdks`.

### Installation
```powershell
python -m pip install pyspark pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

---

## Execution Guide

Run commands from the repository root directory in the following sequence:

### 1. Five-Stage PySpark Data Cleaning Pipeline
```powershell
python code/data-cleaning/01_data_integration.py
python code/data-cleaning/02_data_cleansing.py
python code/data-cleaning/03_data_transformation.py
python code/data-cleaning/04_data_reduction.py
python code/data-cleaning/05_data_analysis_and_visualization.py
```

### 2. Supervised Machine Learning Pipeline
Trains and evaluates 6 supervised algorithms (SLR, MLR, Decision Tree, KNN, Gaussian Naive Bayes, SVM):
```powershell
python code/supervised-ml-algo/main.py
```

### 3. Unsupervised Machine Learning Pipeline
Discovers latent student personas via K-Means ($k=4$) and Mean-Shift density clustering:
```powershell
python code/unsupervised-ml-algo/main.py
```

*Note on Windows:* Warnings regarding `winutils.exe` or `HADOOP_HOME` are standard Spark notifications and do not affect the CSV output generation.

---

## Machine Learning Results Summary

### Supervised ML
- **Simple Linear Regression (SLR):** $R^2 = 0.0446$, evaluating academic comprehension as an initial predictor for modality continuation.
- **Multiple Linear Regression (MLR):** $R^2 = 0.4561$, $RMSE = 0.5723$, demonstrating that combining Academic Learning, Modality Adaptability, Crisis Mitigation, Internet Reliability, and Modality Challenges substantially increases explained variance.
- **Decision Tree Classifier:** Accuracy = **93.84%** (Crisis Mitigation Index represents 66.04% of feature importance).
- **K-Nearest Neighbors (KNN):** Accuracy = **94.40%** ($k=5$), optimal accuracy **95.52%** at $k=12$.
- **Gaussian Naive Bayes:** Accuracy = **92.72%**.
- **Support Vector Machine (SVM):** Accuracy = **94.12%** (RBF kernel, $C=1.0$).

### Unsupervised ML (Personas)
- **K-Means ($k=4$ Personas):**
  1. *Adapted Blended Supporters (27.68%):* High learning index (3.91), high adaptability (3.95), very high support (4.99).
  2. *Balanced Hybrid Adapters (42.12%):* Highest learning index (4.13), strong adaptability (4.20), high support (4.67).
  3. *Crisis-Pragmatic Neutrals (24.95%):* Moderate learning (3.35), acknowledges crisis utility (4.99), high continuation support (4.87).
  4. *Connectivity-Constrained Critics (5.26%):* Lower learning index (2.95), elevated challenges (mean 3.33), lowest support (2.88).
- **Mean-Shift:** Natural density clustering identifying 45 distinct density modes confirming high-adaptability clusters versus connectivity-limited clusters.
