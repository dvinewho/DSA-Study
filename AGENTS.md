# AGENTS.md — DSA Final Project Guide

## Study Information
- **Title**: *Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis*
- **Authors**: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano
- **Institution**: Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines
- **Course**: BSIT 3A – IT 119: Data Scalability and Analytics

## Dataset Architecture (1,530 Raw -> 1,427 Cleaned Records)
1. **Primary Survey (`dataset/Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis.csv`)**: 123 raw -> 117 cleaned records (6 screened out: 1 duplicate email + 5 straight-line responses).
2. **Public S2 (`dataset/public/Le_2026_VN_Blended_Learning_Perceptions/`)**: Le et al. (2026), *Data in Brief* / Mendeley Data DOI `10.17632/tdsspksw83.1` (580 raw -> 508 cleaned, 72 screened out).
3. **Public S3 (`dataset/public/Irwanto_2024_ID_SRL_Blended_Learning/`)**: Irwanto (2024), *Data in Brief* / Mendeley Data DOI `10.17632/7bsvck73ng.1` (577 raw -> 573 cleaned, 4 screened out).
4. **Public S4 (`dataset/public/Makassar_2021_Student_Perception_Online_Learning/`)**: Universitas Negeri Makassar (2021), Mendeley Data DOI `10.17632/8myyf5t45z.3` (250 raw -> 229 cleaned, 21 screened out).

## Execution Order

### 1. Five-Stage PySpark Data Cleaning & Integration Pipeline (`code/data-cleaning/`)
```powershell
python code/data-cleaning/01_data_integration.py
python code/data-cleaning/02_data_cleansing.py
python code/data-cleaning/03_data_transformation.py
python code/data-cleaning/04_data_reduction.py
python code/data-cleaning/05_data_analysis_and_visualization.py
```

### 2. Supervised Machine Learning Suite (`code/supervised-ml-algo/`)
Runs all 6 supervised algorithms (SLR, MLR, Decision Tree, KNN, Gaussian Naive Bayes, SVM):
```powershell
python code/supervised-ml-algo/main.py
```

### 3. Unsupervised Machine Learning Suite (`code/unsupervised-ml-algo/`)
Runs K-Means ($k=4$ personas) and Mean-Shift density clustering:
```powershell
python code/unsupervised-ml-algo/main.py
```

