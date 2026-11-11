"""
03_data_transformation.py
=========================
RUN ORDER: 3 of 5

Stage 3 — Feature Engineering, Composite Indices & NLP Theme Coding (PySpark)
-----------------------------------------------------------------------------
Reads `output/data-cleaning/cleansing/stage2_cleansed_dataset.csv` and performs:
  1. Composite Construct Indices:
       * learning_index = mean(understanding, stay_on_track, academic_performance)
       * adaptability_index = mean(schedule_fit, transition_comfort)
       * crisis_index = mean(reduce_electricity, reduce_fuel, crisis_effectiveness)
       * crisis_learning_gap = crisis_index - learning_index
  2. Multi-select Checkbox Indicators & Challenge Burden:
       * Challenges (ch_*): ch_internet, ch_motivation, ch_interaction, ch_screen_fatigue,
         ch_time_mgmt, ch_commute
       * Improvements (imp_*): imp_tech_support, imp_communication, imp_balance,
         imp_async_materials, imp_scheduling
       * challenge_count = sum(ch_*)
  3. Diagnostic Segmentation:
       * support_category: "Not supportive" (<= 2.5), "Neutral" (2.5 < x < 3.5), "Supportive" (>= 3.5)
       * learning_level: "Low", "Moderate", "High"
       * crisis_level: "Low", "Moderate", "High"
       * gap_typology: "Aligned", "Crisis-Effective but Learning-Struggling",
         "Learning-Adapted beyond Crisis Rating"
  4. Qualitative NLP Keyword Theme Coding for open-ended challenges and suggestions:
       * Challenges (c_*): c_internet_power, c_motivation_focus, c_interaction_consult,
         c_time_workload, c_screen_fatigue, c_commute_cost
       * Suggestions (s_*): s_wifi_tech_support, s_async_lms_materials,
         s_faculty_consultation, s_schedule_f2f_balance, s_workload_breaks
  5. PySpark ML Pipeline (`VectorAssembler`, `StandardScaler`) producing z-standardized scores (*_z).

Outputs -> `output/data-cleaning/transformation/`:
  - `stage3_transformed_dataset.csv` (Transformed unified dataset)
"""

import os
import re

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from pipeline_utils import (
    ALL_11_ITEM_COLS,
    CHALLENGE_CHECKBOX_MAP,
    IMPROVEMENT_CHECKBOX_MAP,
    STAGE2_UNIFIED_CSV,
    STAGE3_UNIFIED_CSV,
    TRANSFORMATION_DIR,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)


def compute_composite_indices(df: DataFrame) -> DataFrame:
    """Compute learning_index, adaptability_index, crisis_index, and crisis_learning_gap."""
    for c in ALL_11_ITEM_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))

    df = (
        df.withColumn(
            "learning_index",
            F.round((F.col("understanding") + F.col("stay_on_track") + F.col("academic_performance")) / 3.0, 3),
        )
        .withColumn(
            "adaptability_index",
            F.round((F.col("schedule_fit") + F.col("transition_comfort")) / 2.0, 3),
        )
        .withColumn(
            "crisis_index",
            F.round((F.col("reduce_electricity") + F.col("reduce_fuel") + F.col("crisis_effectiveness")) / 3.0, 3),
        )
        .withColumn(
            "crisis_learning_gap",
            F.round(F.col("crisis_index") - F.col("learning_index"), 3),
        )
    )
    return df


def encode_multiselect_checkboxes(df: DataFrame) -> DataFrame:
    """One-hot encode multi-select challenges and improvements and compute challenge_count."""
    ch_raw_col = F.coalesce(F.col("challenges_raw"), F.lit(""))
    imp_raw_col = F.coalesce(F.col("improvements_raw"), F.lit(""))

    for col_name, phrase in CHALLENGE_CHECKBOX_MAP.items():
        pat = r"(?i)" + re.escape(phrase)
        df = df.withColumn(col_name, F.when(ch_raw_col.rlike(pat), 1).otherwise(0))

    for col_name, phrase in IMPROVEMENT_CHECKBOX_MAP.items():
        pat = r"(?i)" + re.escape(phrase)
        df = df.withColumn(col_name, F.when(imp_raw_col.rlike(pat), 1).otherwise(0))

    ch_cols = list(CHALLENGE_CHECKBOX_MAP.keys())
    df = df.withColumn("challenge_count", sum(F.col(c) for c in ch_cols).cast("double"))
    return df


def add_behavioral_segmentation(df: DataFrame) -> DataFrame:
    """Add support_category, learning_level, crisis_level, and gap_typology classifications."""
    df = (
        df.withColumn(
            "support_category",
            F.when(F.col("support_continue") <= 2.5, "Not supportive")
            .when(F.col("support_continue") < 3.5, "Neutral")
            .otherwise("Supportive"),
        )
        .withColumn(
            "learning_level",
            F.when(F.col("learning_index") < 2.5, "Low")
            .when(F.col("learning_index") <= 3.75, "Moderate")
            .otherwise("High"),
        )
        .withColumn(
            "crisis_level",
            F.when(F.col("crisis_index") < 2.5, "Low")
            .when(F.col("crisis_index") <= 3.75, "Moderate")
            .otherwise("High"),
        )
        .withColumn(
            "gap_typology",
            F.when(F.abs(F.col("crisis_learning_gap")) <= 0.5, "Aligned")
            .when(F.col("crisis_learning_gap") > 0.5, "Crisis-Effective but Learning-Struggling")
            .otherwise("Learning-Adapted beyond Crisis Rating"),
        )
    )
    return df


def code_qualitative_themes(df: DataFrame) -> DataFrame:
    """Perform keyword theme coding on open-ended text fields using rlike."""
    patterns_c = {
        "c_internet_power": r"(?i)internet|wifi|wi-fi|signal|connection|connectivity|data|power|brownout|blackout|electric|load|unstable",
        "c_motivation_focus": r"(?i)motivat|focus|concentrat|distract|procrastinat|lazy|comprehen|understand|self-study|hard to learn|difficult to absorb",
        "c_interaction_consult": r"(?i)interact|teacher|professor|instructor|classmate|peer|communicat|ask|feedback|discuss|guidance|isolated",
        "c_time_workload": r"(?i)time|schedule|workload|task|deadline|activity|activities|assign|module|overwhelm|manage|balance|household|chores",
        "c_screen_fatigue": r"(?i)screen|eye|fatigue|headache|strain|drain|exhaust|gadget|phone|laptop|health",
        "c_commute_cost": r"(?i)commut|travel|fare|fuel|transport|jeep|bus|distance|boarding|allowance|expense",
    }

    patterns_s = {
        "s_wifi_tech_support": r"(?i)wifi|wi-fi|internet|data|allowance|subsidy|signal|tech|connection|connectivity|free wifi|campus wifi",
        "s_async_lms_materials": r"(?i)material|record|video|module|handout|upload|clear|instruction|lms|asynchronous|resource|guide",
        "s_faculty_consultation": r"(?i)communicat|consult|respond|reply|feedback|teacher|professor|instructor|considerat|explain|teach|monitor",
        "s_schedule_f2f_balance": r"(?i)schedule|balance|f2f|face-to-face|face to face|online|adjust|time|day|lab|major|subject|class",
        "s_workload_breaks": r"(?i)workload|deadline|activity|activities|less|reduce|break|rest|mental|consider",
    }

    c_text = F.coalesce(F.col("challenge_text"), F.lit(""))
    s_text = F.coalesce(F.col("suggestion_text"), F.lit(""))

    for col_name, pat in patterns_c.items():
        df = df.withColumn(
            col_name,
            F.when(c_text.rlike(pat), 1).otherwise(0),
        )

    for col_name, pat in patterns_s.items():
        df = df.withColumn(
            col_name,
            F.when(s_text.rlike(pat), 1).otherwise(0),
        )

    return df


def scale_numerical_features(df: DataFrame) -> DataFrame:
    """Z-standardize the 11 Likert items for PCA dimensionality reduction."""
    assembler = VectorAssembler(inputCols=ALL_11_ITEM_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_z", withMean=True, withStd=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df)
    df_scaled = model.transform(df)

    z_array = vector_to_array(F.col("features_z"))
    for idx, c in enumerate(ALL_11_ITEM_COLS):
        df_scaled = df_scaled.withColumn(f"{c}_z", F.round(z_array[idx], 4))

    return df_scaled.drop("features_raw", "features_z")


def run_stage3_transformation() -> None:
    banner("STAGE 3: TOPIC-ALIGNED FEATURE ENGINEERING & ML SCALING")

    spark = create_spark_session("Stage3_Blended221_Transformation")
    os.makedirs(TRANSFORMATION_DIR, exist_ok=True)

    log(f"Reading Stage 2 Cleansed Dataset: {STAGE2_UNIFIED_CSV}")
    df_clean = read_csv_spark(spark, STAGE2_UNIFIED_CSV)

    # 1. Composite construct indices
    df_indices = compute_composite_indices(df_clean)

    # 2. Multi-select checkboxes & challenge count
    df_checkboxes = encode_multiselect_checkboxes(df_indices)

    # 3. Behavioral segmentation & diagnostic categories
    df_segmented = add_behavioral_segmentation(df_checkboxes)

    # 4. NLP qualitative theme coding
    df_themed = code_qualitative_themes(df_segmented)

    # 5. Standardized scaling for PCA
    df_final = scale_numerical_features(df_themed)

    # Export Stage 3 Transformed Unified CSV
    log(f"Writing Stage 3 Transformed Dataset to: {STAGE3_UNIFIED_CSV}")
    written = write_spark_df_to_csv(df_final, STAGE3_UNIFIED_CSV)
    log(f"Successfully saved {written:,} rows to Stage 3 Transformed Dataset.")

    spark.stop()
    banner("STAGE 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage3_transformation()

