# Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis

**Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, and Divine Mae S. Ponciano**  
*College of Information and Communications Technology / Camarines Norte State College (CNSC)*  
*Daet, Camarines Norte, Philippines*

---

## Abstract

Escalating global fuel prices and seasonal regional power grid strains have prompted Philippine higher education institutions, including Camarines Norte State College (CNSC), to implement a **2-2-1 blended learning schedule** (two days face-to-face, two days synchronous online, and one day asynchronous self-directed learning) as an institutional conservation measure. While the policy reduces campus electricity usage and student commuting costs, its impact on lesson comprehension, academic tracking, and modality satisfaction requires large-scale empirical evaluation. **The primary Philippine survey (123 raw, 117 cleaned records) was combined with three published public datasets (Le et al., 2026; Irwanto, 2024; Universitas Negeri Makassar, 2021), giving 1,530 raw and 1,427 cleaned records for PySpark processing.** Using a five-stage Apache PySpark pipeline (`code/data-cleaning/`), we performed schema harmonization, screening loss auditing (103 invalid records removed), median imputation, composite construct engineering, regular-expression (`rlike`) qualitative theme coding, Principal Component Analysis ($k=3$, explaining $68.22\%$ cumulative variance), and $50\%$ stratified sampling ($n=735$). Results reveal a statistically significant **Crisis–Learning Gap ($\Delta = +1.101$)**: students strongly agree that the 2-2-1 modality mitigates electricity use and household fuel expenses (composite `crisis_index` $= 4.914 / 5.0$), yet report lower lesson understanding and academic tracking (composite `learning_index` $= 3.812 / 5.0$, `adaptability_index` $= 3.828 / 5.0$), alongside a strong preference for face-to-face concentration (`f2f_focus` $= 4.448 / 5.0$). Unsupervised $K$-Means clustering ($k=4$, Silhouette $= 0.3780$) and non-parametric Mean-Shift clustering segment the student body into four actionable personas: *Adapted Blended Supporters* ($27.68\%$), *Balanced Hybrid Adapters* ($42.12\%$), *Crisis-Pragmatic Neutrals* ($24.95\%$), and *Connectivity-Constrained Critics* ($5.26\%$). Across six supervised machine learning models (`code/supervised-ml-algo/`), Multiple Linear Regression explains $45.61\%$ of the variance in support for continuing the 2-2-1 modality ($R^2 = 0.4561, \text{RMSE} = 0.5723$), while optimal $K$-Nearest Neighbors ($95.52\%$ at $k=12$; $94.40\%$ at $k=5$), RBF-Kernel Support Vector Machine ($94.12\%$), Decision Tree ($93.84\%$), and Gaussian Naive Bayes ($92.72\%$) accurately classify student continuation support tiers.

**Keywords:** Blended learning, 2-2-1 modality, power and fuel crisis, Apache PySpark, supervised machine learning, K-Means clustering, higher education, Philippines.

---

## I. Introduction

Higher education institutions (HEIs) in developing countries face increasing operational pressures from macroeconomic energy price volatility and electrical grid constraints. At Camarines Norte State College (CNSC), the **2-2-1 blended learning schedule**—allocating two days to in-person classroom instruction, two days to synchronous online lectures, and one day to asynchronous learning—was adopted to reduce campus electricity consumption and ease student transportation fuel expenses.

Despite its resource-saving rationale, transitioning across three instructional formats within a single academic week imposes substantial cognitive and infrastructural demands on undergraduate students (Garrison & Kanuka, 2004; Zimmerman, 2002). Students residing in rural municipalities or off-campus boarding houses frequently encounter unstable internet connectivity, difficulty staying motivated on asynchronous days, reduced interaction with instructors, and screen fatigue.

To overcome the statistical limitations of small single-campus surveys and fulfill the big-data engineering standards of **IT 119 (Data Scalability and Analytics)**, this study integrates our primary CNSC Google Forms survey ($123$ raw, $117$ cleaned records) with three open-access public datasets on blended and hybrid learning in higher education (Le et al., 2026; Irwanto, 2024; Universitas Negeri Makassar, 2021), yielding **$1,530$ raw and $1,427$ cleaned records** processed through a 5-stage Apache PySpark pipeline, 6 supervised machine learning models, and 2 unsupervised clustering algorithms.

---

## II. Literature Review and Integrated Datasets

1. **Community of Inquiry & Blended Modalities**: Garrison et al. (2000) and Le et al. (2026) demonstrate that blended learning success depends on cognitive presence, instructional design coherence, clear asynchronous learning materials, and sustained faculty-student communication.
2. **Self-Regulated Learning & Digital Equity**: Irwanto (2024) shows that undergraduate self-regulation in blended environments (time management, environment structuring, and goal setting) is strongly moderated by home internet stability and structured task guidance.
3. **Student Perceptions and Modality Support**: The Universitas Negeri Makassar (2021) data confirms that tertiary students value blended schedules for flexible task completion, while maintaining strong preferences for in-person instruction when mastering cognitively demanding topics.

**Integrated Dataset Summary ($1,530$ Raw $\to$ $1,427$ Cleaned Records)**:
- **Primary Study (Chavez, Nañez, & Ponciano, 2026)**: CNSC 2-2-1 Blended Learning Survey ($123$ raw $\to$ $117$ cleaned; $6$ screened out due to $1$ duplicate email and $5$ straight-line responses).
- **Public S2 — Le et al. (2026)**: Vietnam Engineering & Multidisciplinary Universities Blended Learning Survey ($580$ raw $\to$ $508$ cleaned; Mendeley Data DOI: `10.17632/tdsspksw83.1`; *Data in Brief* DOI: `10.1016/j.dib.2026.112958`).
- **Public S3 — Irwanto (2024)**: Indonesia Undergraduate Self-Regulation in Blended Learning Dataset ($577$ raw $\to$ $573$ cleaned; Mendeley Data DOI: `10.17632/7bsvck73ng.1`; *Data in Brief* DOI: `10.1016/j.dib.2024.110066`).
- **Public S4 — Universitas Negeri Makassar (2021)**: Student Perception About Online Learning Dataset ($250$ raw $\to$ $229$ cleaned; Mendeley Data DOI: `10.17632/8myyf5t45z.3`).

---

## III. Methodology

1. **Stage 1 — Multi-Source Integration (`01_data_integration.py`)**: Harmonizes all four real data sources into a canonical 26-column schema (`stage1_integrated_dataset.csv`, $1,530$ rows) and logs per-study screening losses (103 total invalid/careless entries).
2. **Stage 2 — Data Cleansing (`02_data_cleansing.py`)**: Filters `is_valid_record == 1` ($1,427$ retained; $103$ logged in `stage2_cleaning_loss_log.csv`), trims strings, strips placeholder junk tokens, clamps all 11 Likert items to $[1.0, 5.0]$, and applies PySpark `Imputer(strategy="median")`.
3. **Stage 3 — Feature Transformation (`03_data_transformation.py`)**: Computes `learning_index`, `adaptability_index`, `crisis_index`, `crisis_learning_gap`, one-hot checkbox flags (`ch_*`, `imp_*`, `challenge_count`), `support_category`, `gap_typology`, regex open-ended themes (`c_*`, `s_*`), and `StandardScaler` $z$-scores (`*_z`).
4. **Stage 4 — Data Reduction (`04_data_reduction.py`)**: Performs zero-variance pruning, PySpark `PCA(k=3)` ($68.22\%$ cumulative variance explained), and $50\%$ stratified sampling ($n=735$), exporting `integrated_master_dataset.csv` ($1,427 \times 69$).
5. **Stage 5 — Statistical Analysis & ML Modeling (`05_data_analysis_and_visualization.py`, `supervised-ml-algo/`, `unsupervised-ml-algo/`)**: Generates 6 publication figures and executes SLR, MLR, Decision Tree, KNN, Gaussian Naive Bayes, SVM, $K$-Means ($k=4$), and Mean-Shift clustering.

---

## IV. Results and Discussion

### A. Construct Indices and the Crisis–Learning Gap
Across the cleaned master dataset ($N = 1,427$), students rate **Power/Fuel Crisis Mitigation (`crisis_index`, $M = 4.914$)** substantially higher than **Modality Adaptability (`adaptability_index`, $M = 3.828$)** and **Academic Learning Effectiveness (`learning_index`, $M = 3.812$)**, yielding a mean **Crisis–Learning Gap of $+1.101$**. While students overwhelmingly recognize that the 2-2-1 schedule alleviates commuting expenses and campus electricity strain, concentration remains significantly stronger during face-to-face instruction (`f2f_focus`, $M = 4.448$).

### B. Digital Divide and Subgroup Disparities
Students with *Very reliable* home internet achieve higher learning effectiveness and report significantly fewer modality challenges than peers with *Unreliable/No home internet*. Commuting students living with family report the strongest fuel-saving benefit under the 2-2-1 schedule, whereas boarding and dormitory residents prioritize campus Wi-Fi accessibility and evening power stability.

### C. Unsupervised Clustering Personas ($K$-Means & Mean-Shift)
- **$K$-Means ($k=4$, Silhouette $= 0.3780$)**: Identifies four distinct student personas:
  1. *Adapted Blended Supporters* ($27.68\%, n=395$): High learning ($3.910$), high adaptability ($3.949$), and highest support ($4.987$).
  2. *Balanced Hybrid Adapters* ($42.12\%, n=601$): Highest learning ($4.131$), strong adaptability ($4.196$), and high support ($4.672$).
  3. *Crisis-Pragmatic Neutrals* ($24.95\%, n=356$): Solid crisis rating ($4.994$), moderate learning ($3.348$), and high continuation support ($4.865$).
  4. *Connectivity-Constrained Critics* ($5.26\%, n=75$): Low learning ($2.951$), elevated challenges ($3.333$), and lowest continuation support ($2.880$).
- **Mean-Shift ($\text{bandwidth} = 1.8475$, Silhouette $= 0.2931$)**: Non-parametrically segments 45 density modes, identifying dominant supportive clusters ($59.15\%$ and $27.26\%$) and isolating connectivity-compromised subgroups without preset assumptions.

### D. Supervised Machine Learning Evaluation
- **Regression**: Simple Linear Regression (`learning_index` $\to$ `support_continue`) achieves $R^2 = 0.0446$ ($\text{RMSE} = 0.7585, \text{MAE} = 0.4840$, Slope $= 0.3048$). Multiple Linear Regression increases explained variance to **$R^2 = 0.4561$ ($\text{RMSE} = 0.5723, \text{MAE} = 0.3373$)**, with Power/Fuel Crisis Mitigation ($\beta = +0.8259$), Internet Reliability ($\beta = +0.3864$), Academic Learning ($\beta = +0.1183$), and Modality Adaptability ($\beta = +0.1167$) as key predictors.
- **Classification**: Predicting `support_category` (`Supportive` $94.3\%$, `Not supportive` $3.6\%$, `Neutral` $2.0\%$), **$K$-Nearest Neighbors** achieves **$95.52\%$ optimal test accuracy at $k=12$** ($94.40\%$ at $k=5$), followed by **Support Vector Machine (RBF Kernel)** at **$94.12\%$**, **Decision Tree** at **$93.84\%$** (where Crisis Mitigation represents $66.04\%$ of feature importance), and **Gaussian Naive Bayes** at **$92.72\%$**.

---

## V. Conclusion and Recommendations

The 2-2-1 blended learning modality is widely endorsed by college students as an effective response to the power and fuel crisis (`crisis_index` $= 4.914 / 5.0$), garnering continuation support from $94.3\%$ of respondents (`support_continue` $= 4.713 / 5.0$). However, a pronounced Crisis–Learning Gap ($+1.101$) exists, particularly for students facing internet connectivity constraints. To sustain the 2-2-1 schedule effectively, HEIs should:
1. **Expand Campus Wi-Fi and Connectivity Infrastructure** via dedicated study halls and battery/generator backup power for students in transit or off-campus dormitories.
2. **Institutionalize Structured Faculty-Student Consultation** and standardized asynchronous learning modules to sustain academic motivation.
3. **Prioritize Face-to-Face Days for Laboratory, Major, and Computation-Heavy Courses** while designating synchronous/asynchronous sessions for conceptual and general education discussions.
