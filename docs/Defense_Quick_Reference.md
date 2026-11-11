# ORAL DEFENSE QUICK REFERENCE & PANEL Q&A GUIDE

**Study Title:** *Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis*  
**Authors:** Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano  
**Course:** BSIT 3A – IT 119: Data Scalability and Analytics  
**Institution:** Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines  

---

## 1. Study Core Numbers at a Glance (Memory Cheat Sheet)

- **Total Multi-Source Raw Records:** **1,530**
- **Total Validated Cleaned Records:** **1,427** ($93.27\%$ retention)
- **Total Rows Screened Out:** **103** ($6.73\%$)
  - CNSC Primary: 123 raw $\to$ 117 valid (6 screened: 1 duplicate re-submission + 5 straight-line responses with blank text).
  - Le et al. (2026, Vietnam): 580 raw $\to$ 508 valid (72 screened).
  - Irwanto (2024, Indonesia): 577 raw $\to$ 573 valid (4 screened).
  - Universitas Negeri Makassar (2021, Indonesia): 250 raw $\to$ 229 valid (21 screened).
- **Core Construct Means (out of 5.0):**
  - **Power/Fuel Crisis Mitigation Index (`crisis_index`):** **$4.914$** (Very High)
  - **Modality Adaptability Index (`adaptability_index`):** **$3.828$** (Moderate-High)
  - **Academic Learning Index (`learning_index`):** **$3.812$** (Moderate-High)
  - **Face-to-Face Focus Preference (`f2f_focus`):** **$4.448$** (Strong preference for in-person classes)
  - **Support to Continue 2-2-1 (`support_continue`):** **$4.713$** ($94.3\%$ supportive)
  - **The Crisis–Learning Gap ($\Delta = \text{crisis} - \text{learning}$):** **$+1.101$**
- **PySpark PCA Reduction ($k=3$):** Captures **$68.22\%$** cumulative variance ($\text{PC}_1 = 38.76\%$, $\text{PC}_2 = 19.97\%$, $\text{PC}_3 = 9.49\%$).
- **Machine Learning Suite (8 Models):**
  - **MLR ($R^2$):** **$0.4561$** ($\text{RMSE} = 0.5723, \text{MAE} = 0.3373$)
  - **SLR ($R^2$):** **$0.0446$** ($\text{RMSE} = 0.7585, \text{MAE} = 0.4840$)
  - **KNN Accuracy:** **$94.40\%$** at $k=5$ | Peak **$95.52\%$** at $k=12$
  - **SVM Accuracy:** **$94.12\%$** (RBF Kernel, $C=1.0$)
  - **Decision Tree Accuracy:** **$93.84\%$** (Crisis importance = $66.04\%$)
  - **Gaussian Naive Bayes Accuracy:** **$92.72\%$**
  - **K-Means ($k=4$, Silhouette $= 0.3780$):** 4 behavioral personas
  - **Mean-Shift:** 45 natural density modes (Silhouette $= 0.2931$)

---

## 2. Anticipated Panel Questions & High-Scoring Defense Answers

### Q1: "Why did you combine your primary Google Form survey with public datasets?"
> **Answer:**  
> "Our primary campus survey at CNSC gathered 123 raw responses. While valuable for local context, analyzing only 123 rows in manual spreadsheets limits statistical generalizability and would fail the big-data engineering standards of **IT 119: Data Scalability and Analytics**.  
> To test distributed data processing with **Apache PySpark** and train robust supervised and unsupervised machine learning algorithms without overfitting, we integrated our primary survey with three published, peer-reviewed public datasets on blended/online learning (Le et al., 2026; Irwanto, 2024; Universitas Negeri Makassar, 2021). This scaled our sample to **1,530 raw records (1,427 clean records)**, exceeding the 1,000-record threshold while allowing us to evaluate cross-cultural educational dynamics."

---

### Q2: "What is the 'Crisis–Learning Gap' and why is it your study's main contribution?"
> **Answer:**  
> "The Crisis–Learning Gap is the mathematical divergence between how effectively the 2-2-1 modality solves the economic crisis versus how effectively it supports student learning ($\text{crisis\_index} - \text{learning\_index} = +1.101$).  
> While students strongly agree that the 2-2-1 schedule cuts campus electricity consumption ($4.928/5.0$) and saves their families fuel and commuting costs ($4.921/5.0$), their lesson comprehension ($3.812/5.0$) and academic tracking ($3.816/5.0$) are noticeably lower, with students reporting a strong preference for in-person classroom focus ($4.448/5.0$).  
> This proves that institutional policies cannot be evaluated solely on utility bill savings; schools must actively address the pedagogical lag."

---

### Q3: "Why is Multiple Linear Regression (MLR) so much better than Simple Linear Regression (SLR)?"
> **Answer:**  
> "Simple Linear Regression tested whether Academic Learning alone could predict student continuation support, yielding a low $R^2$ of **$0.0446$**.  
> However, when we introduced multiple domain dimensions in MLR (Academic Learning, Schedule Adaptability, Crisis Mitigation, Internet Reliability, F2F Focus, and Challenge Count), our explained variance skyrocketed to **$R^2 = 0.4561$ ($45.61\%$)**.  
> The beta coefficients revealed that the **Power/Fuel Crisis Mitigation Index ($\beta = +0.8259$)** and **Internet Reliability ($\beta = +0.3864$)** are the dominant drivers of student support. This proves that students support 2-2-1 as a multi-dimensional compromise between economic survival and education."

---

### Q4: "How do your Unsupervised Personas help university administrators?"
> **Answer:**  
> "K-Means ($k=4$, Silhouette $= 0.3780$) proved that students are not a homogenous group. It separated the student population into 4 actionable personas:
> 1. **Balanced Hybrid Adapters (42.12%)**: High achievers thriving across all formats.
> 2. **Adapted Blended Supporters (27.68%)**: Moderate-to-high learning with maximum continuation support.
> 3. **Crisis-Pragmatic Neutrals (24.95%)**: Support the policy primarily for fuel/electricity savings while desiring more F2F classes.
> 4. **Connectivity-Constrained Critics (5.26%)**: Students with poor internet, multiple challenges (mean 3.33), and lowest support ($2.88/5.0$).
> Instead of canceling the 2-2-1 schedule or leaving struggling students behind, administrators can directly target interventions (such as solar-powered campus Wi-Fi study hubs) specifically to the 5.26% constrained critics."

---

### Q5: "What data cleansing steps did your PySpark pipeline perform?"
> **Answer:**  
> "In Stage 2 (`02_data_cleansing.py`), PySpark executed automated, auditable screening:
> 1. Removed careless straight-lining responses (zero-variance across all Likert items with blank qualitative feedback) and duplicate submissions, logging 103 screened rows in `stage2_cleaning_loss_log.csv`.
> 2. Normalized whitespace and stripped meaningless junk text tokens (`none`, `n/a`, `wala po`).
> 3. Bounded all Likert columns strictly to $[1.0, 5.0]$.
> 4. Used PySpark's `Imputer(strategy='median')` to impute missing Likert items without distorting ordinal distributions."

---

## 3. Recommended Panel Presentation Flow (5-Minute Script)

1. **Slide 1 — Context & Problem (1 min):** State College crisis response &rarr; 2-2-1 schedule &rarr; Administrative savings vs. Student learning quality.
2. **Slide 2 — Big Data Architecture (1 min):** 1,530 Raw &rarr; 1,427 Clean across 4 sources &rarr; 5-Stage PySpark pipeline &rarr; 68.22% PCA variance explained.
3. **Slide 3 — The Crisis–Learning Gap (1 min):** Mean Crisis ($4.914$) vs. Mean Learning ($3.812$) &rarr; $+1.101$ Gap &rarr; $77.9\%$ F2F focus preference.
4. **Slide 4 — Machine Learning Models (1 min):** MLR $R^2 = 0.4561$ &rarr; KNN $95.52\%$ &rarr; SVM $94.12\%$ &rarr; Decision Tree $93.84\%$ (Crisis 66.04% split importance).
5. **Slide 5 — 4 Personas & Recommendations (1 min):** The 4 K-Means personas &rarr; Actionable policy: Solar Wi-Fi study halls, laboratory scheduling on F2F days, standardized asynchronous modules.

