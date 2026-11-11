"""
02_data_cleansing.py
====================
RUN ORDER: 2 of 5

Stage 2 — Data Cleansing & Invalid Row Screening (PySpark)
----------------------------------------------------------
Reads `output/data-cleaning/integration/stage1_integrated_dataset.csv` and performs:
  1. Filters valid records (`is_valid_record == 1`), removing screened/careless responses
     and exporting `output/data-cleaning/cleansing/stage2_cleaning_loss_log.csv`.
  2. Whitespace trimming & blank-to-NULL normalization across all text fields.
  3. Domain clamping of all 11 quantitative Likert items to [1.0, 5.0].
  4. PySpark `Imputer(strategy="median")` for missing Likert values.
  5. Cleansing of open-ended text (`challenge_text`, `suggestion_text`), removing junk tokens.
  6. Deduplication on `respondent_id`.

Outputs -> `output/data-cleaning/cleansing/`:
  - `stage2_cleansed_dataset.csv` (Cleaned unified dataset: 3,090 valid rows x 24 cols)
  - `stage2_cleaning_loss_log.csv` (Detailed loss audit log)
"""

import os
import re

from pyspark.ml.feature import Imputer
from pyspark.sql import DataFrame, functions as F, types as T

from pipeline_utils import (
    ALL_11_ITEM_COLS,
    CLEANSING_DIR,
    JUNK_TEXT_TOKENS,
    STAGE1_LOST_SUMMARY_CSV,
    STAGE1_UNIFIED_CSV,
    STAGE2_CLEANING_LOG_CSV,
    STAGE2_UNIFIED_CSV,
    banner,
    clamp_likert,
    create_spark_session,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)


def filter_and_log_lost_rows(df: DataFrame, df_loss_summary: DataFrame) -> DataFrame:
    """Filter out invalid/screened rows (`is_valid_record == 0`) and export the loss log."""
    raw_total = df.count()
    write_spark_df_to_csv(df_loss_summary, STAGE2_CLEANING_LOG_CSV)

    valid_df = (
        df.filter(F.col("is_valid_record").cast("int") == 1)
        .drop("is_valid_record", "invalid_reason")
        .dropDuplicates(["respondent_id"])
    )
    clean_total = valid_df.count()
    log(
        f"Stage 2 Screening Filter: {raw_total:,} raw records -> "
        f"{clean_total:,} clean topic-valid records ({raw_total - clean_total:,} screened rows logged)"
    )
    return valid_df


def trim_strings_and_standardize_categories(df: DataFrame) -> DataFrame:
    """Trim strings, convert blanks to null, and standardize demographic numeric fields."""
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
    for c in str_cols:
        trimmed = F.trim(F.col(c))
        df = df.withColumn(c, F.when((trimmed == "") | (trimmed == "null"), None).otherwise(trimmed))

    # Ensure year_num and internet_num are cleanly cast and bounded
    df = df.withColumn(
        "year_num",
        F.when(F.col("year_num").cast("int").between(1, 4), F.col("year_num").cast("int")).otherwise(3),
    ).withColumn(
        "internet_num",
        F.when(F.col("internet_num").cast("double").between(1.0, 3.0), F.col("internet_num").cast("double")).otherwise(2.0),
    )

    return df


def impute_missing_likert_scores(df: DataFrame) -> DataFrame:
    """Clamp Likert items to [1.0, 5.0] and impute any missing scores with column median."""
    for c in ALL_11_ITEM_COLS:
        df = df.withColumn(c, clamp_likert(c))

    imputer = Imputer(
        inputCols=ALL_11_ITEM_COLS,
        outputCols=[f"{c}_imp" for c in ALL_11_ITEM_COLS],
        strategy="median",
    )
    model = imputer.fit(df)
    df_imputed = model.transform(df)

    for c in ALL_11_ITEM_COLS:
        df_imputed = df_imputed.withColumn(c, F.round(F.col(f"{c}_imp"), 3)).drop(f"{c}_imp")

    return df_imputed


def clean_open_ended_text(df: DataFrame) -> DataFrame:
    """Clean open-ended text fields and remove placeholder junk tokens."""
    junk_pattern = r"(?i)^(" + "|".join(re.escape(t) for t in JUNK_TEXT_TOKENS) + r")$"

    for text_col in ["challenge_text", "suggestion_text"]:
        if text_col in df.columns:
            cleaned = F.trim(F.col(text_col))
            df = df.withColumn(
                text_col,
                F.when(cleaned.rlike(junk_pattern) | (cleaned == ""), None).otherwise(cleaned),
            )

    return df


def run_stage2_cleansing() -> None:
    banner("STAGE 2: DATA CLEANSING & SCREENING LOSS AUDIT")

    spark = create_spark_session("Stage2_Blended221_Cleansing")
    os.makedirs(CLEANSING_DIR, exist_ok=True)

    log(f"Reading Stage 1 Unified Dataset: {STAGE1_UNIFIED_CSV}")
    df_raw = read_csv_spark(spark, STAGE1_UNIFIED_CSV)
    df_loss = read_csv_spark(spark, STAGE1_LOST_SUMMARY_CSV)

    # 1. Filter out invalid/screened records
    df_valid = filter_and_log_lost_rows(df_raw, df_loss)

    # 2. Trim strings & standardize demographic fields
    df_trimmed = trim_strings_and_standardize_categories(df_valid)

    # 3. Impute missing Likert scores with median
    df_imputed = impute_missing_likert_scores(df_trimmed)

    # 4. Clean open-ended text fields
    df_clean = clean_open_ended_text(df_imputed)

    # Export Stage 2 Cleansed Unified CSV
    log(f"Writing Stage 2 Cleansed Dataset to: {STAGE2_UNIFIED_CSV}")
    written = write_spark_df_to_csv(df_clean, STAGE2_UNIFIED_CSV)
    log(f"Successfully saved {written:,} rows to Stage 2 Cleansed Dataset.")

    spark.stop()
    banner("STAGE 2 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage2_cleansing()
