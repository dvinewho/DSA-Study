"""
04_data_reduction.py
====================
RUN ORDER: 4 of 5

Stage 4 — Topic-Aligned Data Reduction (PySpark)
------------------------------------------------
Reads `output/data-cleaning/transformation/stage3_transformed_dataset.csv` and performs:
  1. Zero-Variance Feature Pruning (identifies and drops any degenerate constant columns).
  2. Dimensionality Reduction via Principal Component Analysis (PCA k=3)
     on the 11 z-standardized survey items (*_z -> pca_1, pca_2, pca_3).
  3. Numerosity Reduction via 50% Stratified Sampling across support_category (seed=42).

Outputs -> `output/data-cleaning/reduction/`:
  - `integrated_master_dataset.csv` (Final unified master dataset: 3,090 rows)
  - `integrated_master_dataset_sampled50.csv` (50% stratified sample: 1,581 rows)
"""

import os
from typing import List

from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from pipeline_utils import (
    ALL_11_ITEM_COLS,
    ALL_QUALITATIVE_THEME_COLS,
    CHALLENGE_CHECKBOX_MAP,
    CORE_INDEX_COLS,
    IMPROVEMENT_CHECKBOX_MAP,
    MASTER_UNIFIED_CSV,
    PCA_COMPONENTS,
    RANDOM_SEED,
    REDUCTION_DIR,
    SAMPLED_UNIFIED_CSV,
    STAGE3_UNIFIED_CSV,
    STRATIFIED_SAMPLE_FRACTION,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)

Z_SCORE_COLS: List[str] = [f"{c}_z" for c in ALL_11_ITEM_COLS]


def cast_numeric_columns(df: DataFrame) -> DataFrame:
    """Cast numeric item scores, composite indices, checkbox flags, and z-score columns to DoubleType."""
    num_cols = set(
        ALL_11_ITEM_COLS
        + CORE_INDEX_COLS
        + Z_SCORE_COLS
        + ALL_QUALITATIVE_THEME_COLS
        + list(CHALLENGE_CHECKBOX_MAP.keys())
        + list(IMPROVEMENT_CHECKBOX_MAP.keys())
        + ["year_num", "internet_num"]
    )
    for c in num_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
    return df


def prune_zero_variance_features(df: DataFrame) -> DataFrame:
    """Identify and drop any columns with zero variance (countDistinct <= 1)."""
    to_drop = []
    for c in df.columns:
        cnt = df.select(F.countDistinct(c)).first()[0]
        if cnt <= 1:
            to_drop.append(c)

    if to_drop:
        log(f"Dropping zero-variance columns: {to_drop}")
        df = df.drop(*to_drop)
    else:
        log("Zero-variance check: All columns possess valid variance.")
    return df


def apply_pca_reduction(df: DataFrame) -> DataFrame:
    """Compute PCA(k=3) on the 11 z-standardized survey items."""
    log(f"Applying PCA dimensionality reduction (k={PCA_COMPONENTS}) on 11 z-scores...")
    assembler = VectorAssembler(inputCols=Z_SCORE_COLS, outputCol="pca_features")
    df_vector = assembler.transform(df)

    pca = PCA(k=PCA_COMPONENTS, inputCol="pca_features", outputCol="pca_components")
    model = pca.fit(df_vector)
    df_pca = model.transform(df_vector)

    explained_var = [float(round(v, 4)) for v in model.explainedVariance]
    log(f"PCA Explained Variance Ratio: {explained_var} (Cumulative: {round(sum(explained_var), 4)})")

    pca_array = vector_to_array(F.col("pca_components"))
    for i in range(PCA_COMPONENTS):
        df_pca = df_pca.withColumn(f"pca_{i + 1}", F.round(pca_array[i], 4))

    return df_pca.drop("pca_features", "pca_components")


def apply_stratified_sampling(df: DataFrame) -> DataFrame:
    """Perform 50% stratified numerosity reduction sampled across support_category."""
    log(f"Applying {int(STRATIFIED_SAMPLE_FRACTION * 100)}% stratified sampling by support_category...")
    strata = [r[0] for r in df.select("support_category").distinct().collect() if r[0] is not None]
    fractions = {s: STRATIFIED_SAMPLE_FRACTION for s in strata}

    df_sampled = df.stat.sampleBy("support_category", fractions, seed=RANDOM_SEED)
    log(f"Stratified Sampling Result: {df.count():,} master rows -> {df_sampled.count():,} sampled rows.")
    return df_sampled


def run_stage4_reduction() -> None:
    banner("STAGE 4: TOPIC-ALIGNED DATA REDUCTION (PCA & NUMEROSITY)")

    spark = create_spark_session("Stage4_Blended221_Reduction")
    os.makedirs(REDUCTION_DIR, exist_ok=True)

    log(f"Reading Stage 3 Transformed Dataset: {STAGE3_UNIFIED_CSV}")
    df_trans = read_csv_spark(spark, STAGE3_UNIFIED_CSV)
    df_typed = cast_numeric_columns(df_trans)

    # 1. Zero-variance pruning
    df_pruned = prune_zero_variance_features(df_typed)

    # 2. PCA Dimensionality Reduction
    df_master = apply_pca_reduction(df_pruned)

    # 3. 50% Stratified Sampling
    df_sampled = apply_stratified_sampling(df_master)

    # Export Master Unified Dataset (One Big Dataset for Final Output)
    log(f"Writing Master Unified Dataset to: {MASTER_UNIFIED_CSV}")
    written_master = write_spark_df_to_csv(df_master, MASTER_UNIFIED_CSV)
    log(f"Successfully saved {written_master:,} rows to Integrated Master Dataset.")

    # Export Sampled Unified Dataset
    log(f"Writing 50% Stratified Sampled Dataset to: {SAMPLED_UNIFIED_CSV}")
    written_sampled = write_spark_df_to_csv(df_sampled, SAMPLED_UNIFIED_CSV)
    log(f"Successfully saved {written_sampled:,} rows to Sampled Master Dataset.")

    spark.stop()
    banner("STAGE 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage4_reduction()
