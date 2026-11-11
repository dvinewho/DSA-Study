# CHAPTER II: REVIEW OF RELATED LITERATURE AND STUDIES

**Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis**

---

## 2.1 Introduction

This chapter reviews conceptual literature, empirical studies, and open-access educational datasets examining blended and hybrid learning modalities in higher education, student self-regulated learning (SRL), digital infrastructure disparities, and flexible scheduling as an institutional response to energy and fuel crises. It then details the three publicly available research datasets that were harmonized and integrated with our primary Camarines Norte State College (CNSC) survey ($123$ raw $\to$ $117$ cleaned records) to build a unified analytical corpus of **$1,530$ raw entries and $1,427$ cleaned records** for Apache PySpark processing.

---

## 2.2 Related Literature

### 2.2.1 Blended Learning Modalities and the Community of Inquiry Framework
Blended learning—the thoughtful integration of classroom face-to-face learning experiences with online and asynchronous learning experiences (Garrison & Kanuka, 2004)—has evolved from a supplementary pedagogical model into a core operational strategy for higher education institutions. Under the **Community of Inquiry (CoI)** framework (Garrison, Anderson, & Archer, 2000), effective blended learning requires balancing *cognitive presence* (student comprehension and knowledge construction), *social presence* (peer and instructor interaction), and *teaching presence* (instructional design, clear asynchronous materials, and timely feedback). When universities compress face-to-face instruction into two days per week (as in the 2-2-1 schedule), maintaining teaching and social presence during online and asynchronous days becomes the primary determinant of student satisfaction (Le et al., 2026).

### 2.2.2 Flexible Learning as an Institutional Response to Energy and Fuel Crises
Rising global petroleum prices and seasonal strains on national power grids impose dual financial burdens on educational institutions and student households. Campus buildings—particularly computer laboratories, lecture halls, and air-conditioned facilities—represent major institutional electricity consumers. Simultaneously, daily student commuting via public transport or personal motorcycles exposes low- and middle-income families to volatile fuel fares. Shifting a portion of weekly instruction to online and asynchronous delivery reduces campus power demand and cuts weekly commuting trips substantially. However, resource conservation at the institutional level can inadvertently shift electricity and internet data expenses onto student households, creating tension between macroeconomic efficiency and student learning equity.

### 2.2.3 Self-Regulated Learning (SRL) and Digital Divide Constraints
Asynchronous and online learning days place heavy demands on undergraduate students' **self-regulated learning (SRL)** skills, including time management, environment structuring, goal setting, and task persistence (Zimmerman, 2002; Irwanto, 2024). In developing Southeast Asian contexts, these cognitive demands are compounded by the digital divide. Students with unreliable home internet or mobile-data-only connectivity experience frequent disconnections during synchronous lectures, difficulty downloading asynchronous learning materials, heightened screen fatigue, and lower academic self-efficacy (Le et al., 2026).

### 2.2.4 Scalable Educational Data Engineering with Apache PySpark
Traditional educational survey studies often analyze single-campus convenience samples ($N < 150$) using manual spreadsheet tools, limiting statistical generalizability and preventing multi-class machine learning evaluation. **Apache PySpark** provides an in-memory distributed data processing engine capable of harmonizing multi-source survey schemas, executing automated screening audits, imputing isolated missing Likert values via `pyspark.ml.feature.Imputer`, standardizing features via `StandardScaler`, extracting principal components via `PCA`, and coding qualitative text narratives via distributed regular expressions (`rlike`) (Chambers & Zaharia, 2018).

---

## 2.3 Related Studies and Integrated Public Datasets

### 2.3.1 Foreign Studies
- **Le, Le, and Pham (2026) — Blended Learning Perceptions, Engagement, and Academic Achievement in Vietnam**: Published in *Data in Brief* (Vol. 67, 112958; Mendeley Data DOI: `10.17632/tdsspksw83.1`), this study surveyed **$580$ raw ($508$ validated)** undergraduate students across eight multidisciplinary and engineering-oriented universities in Vietnam. Measuring Learning Experience (LE), Instructional Design Quality (IDQ), Cognitive Presence (CP), Student Engagement (SE), Social Presence (SP), and Academic Achievement (AA) on a 5-point Likert scale, the authors demonstrated that instructional design quality and schedule coherence strongly mediate perceived academic achievement.
- **Irwanto (2024) — Undergraduate Self-Regulation in Online and Blended Learning Environments in Indonesia**: Published in *Data in Brief* (Vol. 53, 110066; Mendeley Data DOI: `10.17632/7bsvck73ng.1`), this study examined **$577$ raw ($573$ validated)** undergraduate students at the State University of Jakarta using the Online Self-Regulated Learning Questionnaire (OSLQ, 1–5 Likert scale). The findings revealed that time management (`TM1`) and environment structuring (`ES1`) are severely undermined when students lack reliable home internet or quiet study spaces.
- **Universitas Negeri Makassar (2021) — Student Perception About Online and Blended Learning in Indonesia**: Published on Mendeley Data (DOI: `10.17632/8myyf5t45z.3`), this dataset surveyed **$250$ raw ($229$ validated)** undergraduate students assessing perceptions, study habit adaptation, schedule flexibility, and teaching efficacy in hybrid higher education settings.

### 2.3.2 Local Study (Philippines)
- **Chavez, Nañez, and Ponciano (2026) — Primary CNSC 2-2-1 Blended Learning Survey**: Our primary survey of **$123$ raw ($117$ cleaned)** undergraduate students across multiple colleges at Camarines Norte State College directly evaluates the 2-2-1 schedule implemented in response to the power and fuel crisis, capturing both quantitative Likert ratings and rich qualitative narratives in English and Filipino/Taglish.

---

## 2.4 Synthesis of Integrated Datasets

| Dataset Source | Context & Repository DOI | Raw ($n$) | Cleaned ($n$) | Key Constructs Harmonized into Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **Main Study (Chavez et al., 2026)** | Camarines Norte State College, Philippines (Google Forms Primary Survey) | 123 | 117 | All 11 canonical 2-2-1 Likert items, demographics, internet reliability, checkboxes, and open-ended feedback. |
| **Public S2: Le et al. (2026)** | 8 Universities in Vietnam (*Data in Brief* / Mendeley DOI: `10.17632/tdsspksw83.1`) | 580 | 508 | Learning experience (`LE1`), schedule/design fit (`IDQ2`, `IDQ5`), lesson understanding (`CP4`), engagement (`SE2`), achievement (`AA2`). |
| **Public S3: Irwanto (2024)** | State University of Jakarta, Indonesia (*Data in Brief* / Mendeley DOI: `10.17632/7bsvck73ng.1`) | 577 | 573 | Online Self-Regulated Learning (OSLQ 1–5): time management (`TM1`), environment structuring (`ES1`), task strategies (`TS1`), goal setting (`GS1`), self-evaluation (`SE1`). |
| **Public S4: Universitas Negeri Makassar (2021)** | Universitas Negeri Makassar, Indonesia (Mendeley Data DOI: `10.17632/8myyf5t45z.3`) | 250 | 229 | Online learning delivery satisfaction, task submission comfort, schedule adaptability, in-person vs. online focus preference, continuation support. |
| **TOTAL UNIFIED CORPUS** | **4 Harmonized Studies** | **1,530** | **1,427** | **26 Canonical Stage 1 Columns $\to$ 69 Engineered Master Columns** |

---

## References

- Chambers, B., & Zaharia, M. (2018). *Spark: The definitive guide—Big data processing made simple*. O'Reilly Media.
- Garrison, D. R., Anderson, T., & Archer, W. (2000). Critical inquiry in a text-based environment: Computer conferencing in higher education. *The Internet and Higher Education*, *2*(2–3), 87–105.
- Garrison, D. R., & Kanuka, H. (2004). Blended learning: Uncovering its transformative potential in higher education. *The Internet and Higher Education*, *7*(2), 95–105.
- Irwanto, I. (2024). Data on undergraduate students' self-regulation in online and blended learning environments during the COVID-19 pandemic in Indonesia. *Data in Brief*, *53*, 110066. https://doi.org/10.1016/j.dib.2024.110066 (Dataset DOI: https://doi.org/10.17632/7bsvck73ng.1)
- Le, H. T., Le, H. H., & Pham, T. H. T. (2026). Data on student perceptions of blended learning, engagement, and academic achievement in engineering-related universities in Vietnam. *Data in Brief*, *67*, 112958. https://doi.org/10.1016/j.dib.2026.112958 (Dataset DOI: https://doi.org/10.17632/tdsspksw83.1)
- Universitas Negeri Makassar. (2021). *Dataset Student Perception Online Learning* [Data set]. Mendeley Data. https://doi.org/10.17632/8myyf5t45z.3
- Zimmerman, B. J. (2002). Becoming a self-regulated learner: An overview. *Theory Into Practice*, *41*(2), 64–70.
