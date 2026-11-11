# CHAPTER V: SUMMARY OF FINDINGS, CONCLUSIONS, AND RECOMMENDATIONS

**Study Title:** *Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis*  
**Course:** BSIT 3A – IT 119: Data Scalability and Analytics  
**Researchers:** Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano  
**Institution:** Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines  

---

## 5.1 Summary of Findings

This study evaluated college students' perceptions, academic adaptability, and support for the **2-2-1 blended learning modality** (two days face-to-face, two days synchronous online, and one day asynchronous self-directed learning) implemented as an institutional conservation response to the power and fuel crisis. Using Apache PySpark 4.2.0, the primary survey at Camarines Norte State College ($123$ raw $\to$ $117$ cleaned records) was harmonized with three published, peer-reviewed public datasets (Le et al., 2026; Irwanto, 2024; Universitas Negeri Makassar, 2021) to construct a multi-source master dataset of **$1,530$ raw records** and **$1,427$ validated clean records** ($93.27\%$ retention).

The major empirical findings across the 5-stage big-data pipeline and machine learning suites include:

### 1. Data Cleaning and Screening Loss Audit
Across the integrated dataset, **$103$ invalid records ($6.73\%$)** were identified and screened out using strict Stage 2 rules: $1$ duplicate submission and $5$ straight-line responses with blank qualitative text in the primary survey, $72$ zero-variance responses in Le et al. (2026), $4$ in Irwanto (2024), and $21$ in Makassar (2021). All attrition reasons were logged in `stage2_cleaning_loss_log.csv`.

### 2. The Crisis–Learning Gap
Students rate the **Power/Fuel Crisis Mitigation Index (`crisis_index`, $M = 4.914 / 5.0$)** substantially higher than the **Modality Adaptability Index (`adaptability_index`, $M = 3.828$)** and the **Academic Learning Index (`learning_index`, $M = 3.812$)**, producing an overall **Crisis–Learning Gap of $+1.101$**. While students overwhelmingly agree that the schedule saves campus electricity ($M = 4.928$) and household commuting fuel costs ($M = 4.921$), concentration remains significantly stronger during in-person classroom instruction (`f2f_focus`, $M = 4.448$).

### 3. Policy Continuation Support
Overall endorsement for continuing the 2-2-1 schedule is exceptionally high, averaging **$4.713 / 5.0$**, with **$94.3\%$ ($n = 1,346$)** classified as `Supportive`, $2.0\%$ ($n = 29$) as `Neutral`, and $3.6\%$ ($n = 52$) as `Not supportive`. Bivariate Pearson correlations demonstrate that continuation support is most strongly driven by **Power/Fuel Crisis Mitigation ($r = +0.6164$)**, followed by **Academic Learning ($r = +0.2666$)** and **Modality Adaptability ($r = +0.2228$)**.

### 4. Digital Divide and Subgroup Disparities
Home internet connectivity acts as the primary moderator of student success. Students with *Very reliable* home internet achieve significantly higher learning scores and fewer modality barriers than peers with *Unreliable/No home internet*. Commuting students living with family report the highest fuel savings, whereas boarding house and dormitory residents prioritize campus Wi-Fi access and power stability.

### 5. Supervised Machine Learning Performance
- **Multiple Linear Regression (MLR)** explains **$45.61\%$ of the variance** in continuation support ($R^2 = 0.4561, \text{RMSE} = 0.5723, \text{MAE} = 0.3373$), substantially outperforming Simple Linear Regression ($R^2 = 0.0446$). The strongest positive beta weights are **Crisis Mitigation ($\beta = +0.8259$)** and **Internet Reliability ($\beta = +0.3864$)**.
- **Classification Models**: **$K$-Nearest Neighbors** achieved the highest test accuracy at **$95.52\%$ ($k=12$)** ($94.40\%$ at $k=5$), followed by **Support Vector Machine (RBF Kernel, $C=1.0$)** at **$94.12\%$**, **Decision Tree Classifier** at **$93.84\%$** (where Crisis Mitigation represents $66.04\%$ of feature importance), and **Gaussian Naive Bayes** at **$92.72\%$**.

### 6. Unsupervised Personas Discovery
- **$K$-Means Clustering ($k=4$, Silhouette $= 0.3780$)** segmented the student body into four distinct behavioral personas:
  1. *Balanced Hybrid Adapters* ($42.12\%, n=601$): Highest learning ($4.131$), strong adaptability ($4.196$), and high continuation support ($4.672$).
  2. *Adapted Blended Supporters* ($27.68\%, n=395$): High learning ($3.910$), high adaptability ($3.949$), and highest support ($4.987$).
  3. *Crisis-Pragmatic Neutrals* ($24.95\%, n=356$): Strong crisis utility awareness ($4.994$), moderate learning ($3.348$), and solid support ($4.865$).
  4. *Connectivity-Constrained Critics* ($5.26\%, n=75$): Low learning ($2.951$), highest challenge burden ($3.333$), and lowest continuation support ($2.880$).
- **Mean-Shift Clustering** discovered 45 natural density modes (Silhouette $= 0.2931$), confirming that the vast majority belong to large supportive density cores while isolating vulnerable connectivity-constrained outliers without preset assumptions.

---

## 5.2 Conclusions

Based on the empirical findings, the following conclusions are drawn:

1. **Macroeconomic and Resource Effectiveness**: The 2-2-1 blended learning schedule achieves its core institutional objective as a crisis-mitigation mechanism. College students strongly recognize and appreciate its role in lowering campus electrical loads and alleviating daily commuting and petroleum expenditures.
2. **The Pedagogical Reality of the Crisis–Learning Gap**: While the economic rationale is clear, learning comprehension and academic tracking lag behind resource-saving satisfaction. Face-to-face instruction remains the preferred and most effective setting for complex cognitive mastery.
3. **Connectivity Equity as the Decisive Linchpin**: Student dissatisfaction and resistance to the 2-2-1 schedule are not philosophical objections to blended learning, but practical consequences of unstable internet access, power interruptions, and screen fatigue.
4. **Predictive Capability of Scalable Machine Learning**: Machine learning algorithms reliably model student sentiment. The multi-dimensional regression models prove that continuation support depends jointly on macroeconomic utility and pedagogical adaptability, while classifiers can automatically detect students at risk of falling into non-supportive or academically struggling cohorts.
5. **Actionable Personas for Targeted Intervention**: The four discovered personas demonstrate that students do not experience blended learning uniformly. Institutional policies must distinguish between thriving hybrid adapters and connectivity-constrained learners requiring targeted institutional support.

---

## 5.3 Recommendations

To maximize the benefits of the 2-2-1 blended schedule while actively closing the Crisis–Learning Gap, the following recommendations are offered:

### 1. For University Administrators and Campus Planners
- **Establish Solar-Powered Connectivity Hubs**: Provide dedicated, quiet study spaces on campus equipped with high-speed Wi-Fi, charging outlets, and emergency battery/solar backup power for commuting and dormitory students during online/asynchronous days.
- **Optimize Schedule Distribution**: Ensure that laboratory sessions, major computing/engineering subjects, and heavy computation courses are strictly scheduled during the two face-to-face days, reserving conceptual and discussion-heavy topics for online days.
- **Subsidize Academic Data Connectivity**: Partner with telecommunications providers to provide educational data allowances or offline-accessible learning tablets for underprivileged students residing in low-coverage rural areas.

### 2. For Faculty Members and Instructional Designers
- **Standardize Asynchronous Learning Materials**: Deliver asynchronous course modules in lightweight, pre-downloadable formats (PDF guides, compressed audio summaries, low-bandwidth video clips) accompanied by explicit rubrics and weekly task checklists.
- **Institutionalize Synchronous Consultation Windows**: Designate regular, low-stakes virtual office hours or messaging channels to sustain teacher presence, provide timely feedback, and counter student isolation during remote days.
- **Manage Screen Fatigue and Cognitive Workload**: Avoid excessive consecutive screen hours by embedding active off-screen problem-solving tasks, independent reading, and structured micro-breaks into synchronous lecture days.

### 3. For College Students
- **Cultivate Self-Regulated Learning (SRL) Habits**: Establish a dedicated study routine, maintain a personal academic calendar for submission deadlines, and proactively minimize domestic distractions on asynchronous days.
- **Form Collaborative Peer Study Cohorts**: Form small study circles with classmates living nearby to share internet connectivity resources, discuss challenging coursework, and sustain peer motivation during non-classroom days.

### 4. For Future Data Science and Educational Researchers
- **Incorporate Objective Utility Metrics**: Complement self-reported Likert surveys with empirical campus electricity billing records, kilowatt-hour meter readings, and student transportation fare receipts.
- **Conduct Longitudinal Academic Tracking**: Track student grade point averages (GPAs) and retention rates across semesters before, during, and after blended learning interventions to measure causal learning effects.
- **Expand Multi-Campus Comparative Analytics**: Scale the Apache PySpark pipeline to integrate survey data across diverse state universities and colleges (SUCs) across all regions of the Philippines.
