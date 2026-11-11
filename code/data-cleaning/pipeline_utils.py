"""
pipeline_utils.py
=================
Shared utilities, SparkSession factory, directory paths, domain constants,
data cleaning rules, and Windows-safe CSV writers for the 5-stage DSA Final
PySpark Multi-Source Pipeline.

Main Study:
  "Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis"
  Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano
  Institution: Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines

Integrated Public Studies:
  - Public S2: Hoai Thu Le, Hieu Hoc Le, & Thanh Hai Thi Pham (2026),
    "Data on student perceptions of blended learning, engagement, and academic achievement
    in engineering-related universities in Vietnam"
    (Data in Brief / Mendeley Data DOI: 10.17632/tdsspksw83.1; 10.1016/j.dib.2026.112958)
  - Public S3: Luis Matosas-López (2025), Rey Juan Carlos University,
    "Dataset on student evaluations of a BARS questionnaire designed for blended learning
    teaching modalities" (Zenodo DOI: 10.5281/zenodo.15160903)
  - Public S4: Irwanto Irwanto (2024), State University of Jakarta,
    "Data on undergraduate students' self-regulation in online and blended learning
    environments" (Data in Brief / Mendeley Data DOI: 10.17632/7bsvck73ng.1)
  - Public S5: Francisco B. Valdespinosa Jr. (2026), University of Perpetual Help System DALTA,
    "Hybrid Learning in Higher Education: An Analysis of Online and Face-to-Face Instruction"
    (Zenodo DOI: 10.5281/zenodo.22154143)

Canonical 11 Likert Items (1.0 = Lowest/Strongly Disagree to 5.0 = Highest/Strongly Agree):
  * overall_experience   [1. Overall experience with the 2-2-1 blended learning modality]
  * schedule_fit         [2. Fits well with daily schedule and responsibilities]
  * transition_comfort   [3. Comfortable transitioning between F2F, online, and async days]
  * understanding        [4. Positively affected understanding of lessons]
  * stay_on_track        [5. Able to complete requirements and stay on track academically]
  * f2f_focus            [6. Focus and learning are stronger on F2F days than online/async]
  * academic_performance [7. Perceived effect on academic performance vs. full F2F]
  * reduce_electricity   [8. Helped reduce electricity consumption at school]
  * reduce_fuel          [9. Helped reduce household fuel/transportation expenses]
  * crisis_effectiveness [10. Overall effectiveness in addressing the power and fuel crisis]
  * support_continue     [11. Supportive of continuing the 2-2-1 modality in current form]
"""

import csv
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import pyspark
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

# Configure Windows paths and workers
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Auto-resolve valid JAVA_HOME on Windows if current JAVA_HOME is missing or invalid
_cur_java = os.environ.get("JAVA_HOME", "")
if not _cur_java or not os.path.exists(os.path.join(_cur_java, "bin", "java.exe")):
    _jdks_dir = os.path.join(os.path.expanduser("~"), ".jdks")
    if os.path.isdir(_jdks_dir):
        for _cand in sorted(os.listdir(_jdks_dir), reverse=True):
            _cand_path = os.path.join(_jdks_dir, _cand)
            if os.path.exists(os.path.join(_cand_path, "bin", "java.exe")):
                os.environ["JAVA_HOME"] = _cand_path
                os.environ["PATH"] = os.path.join(_cand_path, "bin") + os.pathsep + os.environ.get("PATH", "")
                break

# ==============================================================================
# DIRECTORY & FILE PATH CONSTANTS
# ==============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
PUBLIC_DATASET_DIR = os.path.join(DATASET_DIR, "public")

OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output", "data-cleaning")
INTEGRATION_DIR = os.path.join(OUTPUT_ROOT, "integration")
STAGING_DIR = os.path.join(INTEGRATION_DIR, "staging")
CLEANSING_DIR = os.path.join(OUTPUT_ROOT, "cleansing")
TRANSFORMATION_DIR = os.path.join(OUTPUT_ROOT, "transformation")
REDUCTION_DIR = os.path.join(OUTPUT_ROOT, "reduction")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "analysis")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")

# Unified dataset handoff paths across the 5 stages
STAGE1_UNIFIED_CSV = os.path.join(INTEGRATION_DIR, "stage1_integrated_dataset.csv")
STAGE1_LOST_SUMMARY_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_by_study.csv")
STAGE1_LOST_REASONS_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_reasons_detail.csv")

STAGE2_UNIFIED_CSV = os.path.join(CLEANSING_DIR, "stage2_cleansed_dataset.csv")
STAGE2_CLEANING_LOG_CSV = os.path.join(CLEANSING_DIR, "stage2_cleaning_loss_log.csv")

STAGE3_UNIFIED_CSV = os.path.join(TRANSFORMATION_DIR, "stage3_transformed_dataset.csv")

MASTER_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset.csv")
SAMPLED_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset_sampled50.csv")

ANALYSIS_LOST_ROWS_CSV = os.path.join(ANALYSIS_DIR, "study_rows_lost_and_reasons.csv")

# Input files
MAIN_STUDY_CSV = os.path.join(
    DATASET_DIR,
    "Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis.csv",
)
LE_2026_XLSX = os.path.join(
    PUBLIC_DATASET_DIR,
    "Le_2026_VN_Blended_Learning_Perceptions",
    "Blended learning_data set.xlsx",
)
IRWANTO_2024_XLSX = os.path.join(
    PUBLIC_DATASET_DIR,
    "Irwanto_2024_ID_SRL_Blended_Learning",
    "Data for SPSS.xlsx",
)
MAKASSAR_2021_XLSX = os.path.join(
    PUBLIC_DATASET_DIR,
    "Makassar_2021_Student_Perception_Online_Learning",
    "Dataset Student Perception Online Learning.xlsx",
)

# ==============================================================================
# DOMAIN & PIPELINE CONSTANTS
# ==============================================================================
LIKERT_MIN = 1.0
LIKERT_MAX = 5.0
LIKERT_NEUTRAL = 3.0

PCA_COMPONENTS = 3
STRATIFIED_SAMPLE_FRACTION = 0.50
RANDOM_SEED = 42

ALL_11_ITEM_COLS: List[str] = [
    "overall_experience",
    "schedule_fit",
    "transition_comfort",
    "understanding",
    "stay_on_track",
    "f2f_focus",
    "academic_performance",
    "reduce_electricity",
    "reduce_fuel",
    "crisis_effectiveness",
    "support_continue",
]

ITEM_LABELS: Dict[str, str] = {
    "overall_experience": "Overall experience with 2-2-1",
    "schedule_fit": "Fits my schedule/responsibilities",
    "transition_comfort": "Comfortable switching F2F/online/async",
    "understanding": "Improved my understanding of lessons",
    "stay_on_track": "Can complete requirements, stay on track",
    "f2f_focus": "Focus is stronger on F2F days",
    "academic_performance": "Academic performance vs. full F2F",
    "reduce_electricity": "Reduced electricity use at school",
    "reduce_fuel": "Reduced my household fuel expenses",
    "crisis_effectiveness": "Effective vs. power/fuel crisis",
    "support_continue": "Support continuing 2-2-1 as is",
}

CORE_INDEX_COLS: List[str] = [
    "learning_index",
    "adaptability_index",
    "crisis_index",
    "crisis_learning_gap",
    "challenge_count",
]

# Multi-select Challenge & Improvement Checkboxes
CHALLENGE_CHECKBOX_MAP: Dict[str, str] = {
    "ch_internet": "Internet connectivity issues",
    "ch_motivation": "Difficulty staying motivated during asynchronous days",
    "ch_interaction": "Lack of interaction with teachers/classmates",
    "ch_screen_fatigue": "Increased screen fatigue",
    "ch_time_mgmt": "Difficulty managing time across formats",
    "ch_commute": "Increased travel/commute burden on face-to-face days",
}

IMPROVEMENT_CHECKBOX_MAP: Dict[str, str] = {
    "imp_tech_support": "Internet/technical support",
    "imp_communication": "Teacher-student communication",
    "imp_balance": "Balance between online and face-to-face days",
    "imp_async_materials": "Asynchronous learning materials",
    "imp_scheduling": "Class scheduling",
}

# Qualitative Open-Ended Theme binary columns (rlike keyword coding)
CHALLENGE_THEME_COLS: List[str] = [
    "c_internet_power",
    "c_motivation_focus",
    "c_interaction_consult",
    "c_time_workload",
    "c_screen_fatigue",
    "c_commute_cost",
]

SUGGESTION_THEME_COLS: List[str] = [
    "s_wifi_tech_support",
    "s_async_lms_materials",
    "s_faculty_consultation",
    "s_schedule_f2f_balance",
    "s_workload_breaks",
]

ALL_QUALITATIVE_THEME_COLS: List[str] = CHALLENGE_THEME_COLS + SUGGESTION_THEME_COLS

JUNK_TEXT_TOKENS: List[str] = [
    "n/a", "na", "none", "no", "nothing", "nil", "unsure",
    "not sure", "-", ".", "?", "idk", "null", "nan", "wala",
    "wala po", "wala naman po", "none po", "no suggestion",
    "no suggestions", "none so far", "none for now", "2", "4",
]

UNIFIED_CANONICAL_SCHEMA: List[str] = [
    "respondent_id",
    "source_study",
    "study_role",
    "program_group",
    "year_level",
    "year_num",
    "residence_group",
    "internet_group",
    "internet_num",
    "overall_experience",
    "schedule_fit",
    "transition_comfort",
    "understanding",
    "stay_on_track",
    "f2f_focus",
    "academic_performance",
    "reduce_electricity",
    "reduce_fuel",
    "crisis_effectiveness",
    "support_continue",
    "challenges_raw",
    "improvements_raw",
    "challenge_text",
    "suggestion_text",
    "is_valid_record",
    "invalid_reason",
]


def banner(title: str) -> None:
    """Print a formatted stage section header."""
    line = "=" * 90
    print(f"\n{line}\n  {title}\n{line}")


def log(msg: str) -> None:
    """Print an indented pipeline status line."""
    print(f"  [Blended-221-Pipeline] {msg}")


def create_spark_session(app_name: str, shuffle_partitions: int = 8) -> SparkSession:
    """Create and configure a Windows-compatible PySpark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_csv_spark(spark: SparkSession, path: str, infer_schema: bool = False) -> DataFrame:
    """Read a UTF-8 CSV file into a Spark DataFrame with multi-line quote support."""
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true" if infer_schema else "false")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(path)
    )


def write_spark_df_to_csv(
    df: DataFrame,
    out_path: str,
    exclude_cols: Optional[Iterable[str]] = None,
) -> int:
    """
    Stream a Spark DataFrame to a single UTF-8-SIG CSV file via `toLocalIterator()`.
    Bypasses Hadoop `winutils.exe` on Windows while keeping driver memory bounded.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    exclude = set(exclude_cols or [])
    scalar_cols = [
        f.name
        for f in df.schema.fields
        if f.name not in exclude
        and not isinstance(f.dataType, (T.ArrayType, T.MapType, T.StructType))
    ]

    selected_df = df.select(scalar_cols)
    field_names = [f.name for f in selected_df.schema.fields]

    count = 0
    with open(out_path, mode="w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(field_names)
        for row in selected_df.toLocalIterator():
            row_dict = row.asDict()
            writer.writerow([row_dict.get(c) for c in field_names])
            count += 1

    return count


def clamp_likert(col_name: str) -> F.Column:
    """Clamp numeric score to [LIKERT_MIN, LIKERT_MAX]."""
    c = F.col(col_name).cast("double")
    return (
        F.when(c.isNull(), None)
        .when(c < LIKERT_MIN, F.lit(LIKERT_MIN))
        .when(c > LIKERT_MAX, F.lit(LIKERT_MAX))
        .otherwise(F.round(c, 3))
    )
