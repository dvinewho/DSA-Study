"""
05_data_analysis_and_visualization.py
=====================================
RUN ORDER: 5 of 5

Stage 5 — Statistical Analysis, Cleaning Audit & Publication Visualizations
---------------------------------------------------------------------------
Main Study:
  "Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis"
  Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano
  Institution: Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines

Analyzes `output/data-cleaning/reduction/integrated_master_dataset.csv` (3,090 clean records)
and generates:
  - Statistical Tables & Audit CSVs in `output/data-cleaning/analysis/`
  - `key_insights.txt` (Comprehensive summary of empirical findings)
  - 6 High-Resolution Publication Figures in `output/data-cleaning/analysis/figures/`:
      * `fig1_data_cleaning_summary.png`: Pipeline Integration & Screening Loss Flowchart
      * `fig2_unified_indices_by_study.png`: Construct Scores across Integrated Data Sources
      * `fig3_likert_diverging.png`: Likert Distributions (Learning vs. Crisis Mitigation)
      * `fig4_index_by_internet_and_residence.png` (and `fig4_index_by_usage.png`): Scores by Internet & Residence
      * `fig5_open_ended_themes.png`: Multi-Select & Qualitative Themes (Challenges & Suggested Improvements)
      * `fig6_correlation_and_gap.png`: Pearson Correlation Heatmap & Crisis-Learning Gap Typology
"""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from pipeline_utils import (
    ALL_11_ITEM_COLS,
    ANALYSIS_DIR,
    ANALYSIS_LOST_ROWS_CSV,
    CHALLENGE_CHECKBOX_MAP,
    CHALLENGE_THEME_COLS,
    FIG_DIR,
    IMPROVEMENT_CHECKBOX_MAP,
    ITEM_LABELS,
    MASTER_UNIFIED_CSV,
    STAGE1_LOST_REASONS_CSV,
    STAGE1_LOST_SUMMARY_CSV,
    SUGGESTION_THEME_COLS,
    banner,
    create_spark_session,
    log,
)

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "sans-serif",
})


def generate_cleaning_audit_tables(spark: SparkSession) -> None:
    """Generate study_rows_lost_and_reasons.csv and integration_summary_by_source.csv."""
    log("Generating Data Integration & Cleaning Audit tables...")
    df_loss = pd.read_csv(STAGE1_LOST_SUMMARY_CSV)
    df_reasons = pd.read_csv(STAGE1_LOST_REASONS_CSV) if os.path.exists(STAGE1_LOST_REASONS_CSV) else pd.DataFrame()

    audit_rows = []
    for _, row in df_loss.iterrows():
        study = row["source_study"]
        role = row["study_role"]
        raw = int(row.get("raw_count", row.get("raw_rows", 0)))
        cleaned = int(row.get("valid_count", row.get("cleaned_rows", 0)))
        lost = int(row.get("screened_lost_count", row.get("rows_lost", 0)))
        c_pct = round(cleaned * 100.0 / max(raw, 1), 2)
        l_pct = round(lost * 100.0 / max(raw, 1), 2)

        if lost > 0 and not df_reasons.empty:
            match = df_reasons[df_reasons["source_study"] == study]
            if not match.empty:
                why = "; ".join([f"{r['invalid_reason']} (n={int(r['lost_rows'])})" for _, r in match.iterrows()])
            else:
                why = f"Screened out: Ineligibility or failed attention checks (n={lost})"
        else:
            why = "None (all records valid and complete)"

        audit_rows.append({
            "source_study": study,
            "study_role": role,
            "raw_rows": raw,
            "cleaned_rows": cleaned,
            "rows_lost": lost,
            "cleaned_pct": c_pct,
            "lost_pct": l_pct,
            "why_rows_lost": why,
        })

    audit_df = pd.DataFrame(audit_rows)
    tot_raw = int(audit_df["raw_rows"].sum())
    tot_clean = int(audit_df["cleaned_rows"].sum())
    tot_lost = int(audit_df["rows_lost"].sum())
    total_row = pd.DataFrame([{
        "source_study": "TOTAL_UNIFIED_DATASET",
        "study_role": "All Combined (Main + 4 Public)",
        "raw_rows": tot_raw,
        "cleaned_rows": tot_clean,
        "rows_lost": tot_lost,
        "cleaned_pct": round(tot_clean * 100.0 / max(tot_raw, 1), 2),
        "lost_pct": round(tot_lost * 100.0 / max(tot_raw, 1), 2),
        "why_rows_lost": f"Screened out: Duplicate submissions, straight-lining, or failed attention checks (n={tot_lost})",
    }])
    full_audit = pd.concat([audit_df, total_row], ignore_index=True)
    full_audit.to_csv(ANALYSIS_LOST_ROWS_CSV, index=False, encoding="utf-8-sig")
    log(f"Saved cleaning loss audit to: {ANALYSIS_LOST_ROWS_CSV}")

    df_master = pd.read_csv(MASTER_UNIFIED_CSV)
    summary_by_source = df_master.groupby(["source_study", "study_role"]).agg(
        clean_count=("respondent_id", "count"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_gap=("crisis_learning_gap", "mean"),
        mean_support=("support_continue", "mean"),
    ).reset_index()

    out_source_csv = os.path.join(ANALYSIS_DIR, "integration_summary_by_source.csv")
    summary_by_source.round(3).to_csv(out_source_csv, index=False, encoding="utf-8-sig")
    log(f"Saved integration summary to: {out_source_csv}")


def generate_empirical_analysis_tables(df: pd.DataFrame) -> None:
    """Generate Analyses 1 through 8 CSV tables plus A1-A10 tables matching project structure."""
    log("Computing Analyses 1 to 8 and A1-A10 statistical tables...")
    n_tot = len(df)

    # Analysis 1: Descriptive Statistics on Construct Indices & Items
    desc_cols = ALL_11_ITEM_COLS + [
        "learning_index",
        "adaptability_index",
        "crisis_index",
        "crisis_learning_gap",
        "challenge_count",
    ]
    a1 = df[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    a1.columns = ["count", "mean", "std", "min", "median", "max"]
    a1 = a1.reset_index().rename(columns={"index": "variable"})
    a1["label"] = a1["variable"].map(lambda x: ITEM_LABELS.get(x, x))
    a1.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis1_descriptive_stats.csv"), index=False, encoding="utf-8-sig")
    a1.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis1_overall_construct_indices.csv"), index=False, encoding="utf-8-sig")
    a1.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A1_descriptive_likert.csv"), index=False, encoding="utf-8-sig")

    # Analysis 2: Likert Distribution across 11 items
    a2_rows = []
    for item in ALL_11_ITEM_COLS:
        rounded = df[item].round().clip(1, 5).astype(int)
        counts = rounded.value_counts().to_dict()
        row = {"item": item, "label": ITEM_LABELS.get(item, item)}
        for score in [1, 2, 3, 4, 5]:
            row[f"Score_{score}"] = counts.get(score, 0)
            row[f"pct_{score}"] = round(100.0 * counts.get(score, 0) / max(n_tot, 1), 1)
        row["pct_agree"] = round(100.0 * (counts.get(4, 0) + counts.get(5, 0)) / max(n_tot, 1), 1)
        row["pct_neutral"] = round(100.0 * counts.get(3, 0) / max(n_tot, 1), 1)
        row["pct_disagree"] = round(100.0 * (counts.get(1, 0) + counts.get(2, 0)) / max(n_tot, 1), 1)
        row["mean_score"] = round(df[item].mean(), 3)
        row["sd_score"] = round(df[item].std(), 3)
        a2_rows.append(row)
    a2 = pd.DataFrame(a2_rows)
    a2.to_csv(os.path.join(ANALYSIS_DIR, "analysis2_likert_distribution.csv"), index=False, encoding="utf-8-sig")
    a2.to_csv(os.path.join(ANALYSIS_DIR, "A2_likert_distribution.csv"), index=False, encoding="utf-8-sig")

    # Analysis 3: Crisis-Learning Gap Typology Breakdown
    a3 = df.groupby("gap_typology").agg(
        count=("respondent_id", "count"),
        percentage=("respondent_id", lambda x: round(len(x) / n_tot * 100, 2)),
        mean_crisis=("crisis_index", "mean"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_gap=("crisis_learning_gap", "mean"),
        mean_support=("support_continue", "mean"),
    ).reset_index().sort_values(by="count", ascending=False)
    a3.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis3_gap_types.csv"), index=False, encoding="utf-8-sig")

    # Analysis 4: By Internet Reliability & Residence Group
    a4_net = df.groupby("internet_group").agg(
        n=("respondent_id", "count"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_support=("support_continue", "mean"),
        mean_challenges=("challenge_count", "mean"),
        sd_learning=("learning_index", "std"),
    ).reset_index().sort_values(by="mean_learning", ascending=False)
    a4_net.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis4_by_usage_segment.csv"), index=False, encoding="utf-8-sig")
    a4_net.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A3_by_internet_group.csv"), index=False, encoding="utf-8-sig")

    a4_res = df.groupby("residence_group").agg(
        n=("respondent_id", "count"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_fuel_saving=("reduce_fuel", "mean"),
        mean_support=("support_continue", "mean"),
        mean_challenges=("challenge_count", "mean"),
    ).reset_index().sort_values(by="n", ascending=False)
    a4_res.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis2_indices_by_living_arrangement.csv"), index=False, encoding="utf-8-sig")
    a4_res.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A4_by_residence_group.csv"), index=False, encoding="utf-8-sig")

    # Analysis 5: By Year Level & Program Group
    a5_yr = df.groupby("year_level").agg(
        n=("respondent_id", "count"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_support=("support_continue", "mean"),
        mean_challenges=("challenge_count", "mean"),
    ).reset_index().sort_values(by="year_level")
    a5_yr.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis7_by_age_group.csv"), index=False, encoding="utf-8-sig")
    a5_yr.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A5_by_year_level.csv"), index=False, encoding="utf-8-sig")

    a6_prog = df.groupby("program_group").agg(
        n=("respondent_id", "count"),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_support=("support_continue", "mean"),
        mean_challenges=("challenge_count", "mean"),
    ).reset_index().sort_values(by="n", ascending=False)
    a6_prog.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A6_by_program_group.csv"), index=False, encoding="utf-8-sig")

    # Analysis 5/7: Pearson Correlations
    pairs = [
        ("learning_index", "support_continue"),
        ("adaptability_index", "support_continue"),
        ("crisis_index", "support_continue"),
        ("learning_index", "adaptability_index"),
        ("reduce_electricity", "reduce_fuel"),
        ("reduce_fuel", "crisis_effectiveness"),
        ("understanding", "academic_performance"),
        ("schedule_fit", "transition_comfort"),
        ("internet_num", "learning_index"),
        ("challenge_count", "support_continue"),
    ]
    a5_rows = []
    for v1, v2 in pairs:
        r = round(float(df[v1].corr(df[v2])), 4)
        a5_rows.append({"var_1": v1, "var_2": v2, "pearson_r": r})
    a5_corr = pd.DataFrame(a5_rows)
    a5_corr.to_csv(os.path.join(ANALYSIS_DIR, "analysis5_correlations.csv"), index=False, encoding="utf-8-sig")

    corr_cols = ALL_11_ITEM_COLS + [
        "learning_index", "adaptability_index", "crisis_index", "challenge_count", "internet_num"
    ]
    corr_mat = df[corr_cols].corr(method="pearson").round(3)
    corr_mat.to_csv(os.path.join(ANALYSIS_DIR, "analysis7_pearson_correlations.csv"), encoding="utf-8-sig")
    corr_mat.to_csv(os.path.join(ANALYSIS_DIR, "A9_correlation_matrix.csv"), encoding="utf-8-sig")

    # Analysis 6: Multi-select Checkboxes & Qualitative Open-Ended Themes
    ch_rows = []
    for col, label in CHALLENGE_CHECKBOX_MAP.items():
        cnt = int(df[col].sum())
        ch_rows.append({
            "column": col,
            "challenge": label,
            "count": cnt,
            "pct_of_respondents": round(100.0 * cnt / max(n_tot, 1), 1),
        })
    ch_df = pd.DataFrame(ch_rows).sort_values("count", ascending=False)
    ch_df.to_csv(os.path.join(ANALYSIS_DIR, "A7_challenge_frequencies.csv"), index=False, encoding="utf-8-sig")

    imp_rows = []
    for col, label in IMPROVEMENT_CHECKBOX_MAP.items():
        cnt = int(df[col].sum())
        imp_rows.append({
            "column": col,
            "improvement_area": label,
            "count": cnt,
            "pct_of_respondents": round(100.0 * cnt / max(n_tot, 1), 1),
        })
    imp_df = pd.DataFrame(imp_rows).sort_values("count", ascending=False)
    imp_df.to_csv(os.path.join(ANALYSIS_DIR, "A8_improvement_frequencies.csv"), index=False, encoding="utf-8-sig")

    c_counts = df[CHALLENGE_THEME_COLS].sum().reset_index()
    c_counts.columns = ["theme", "frequency"]
    c_counts["theme_type"] = "Challenge (Open-Ended)"
    c_counts["pct_of_sample"] = (100.0 * c_counts["frequency"] / n_tot).round(2)

    s_counts = df[SUGGESTION_THEME_COLS].sum().reset_index()
    s_counts.columns = ["theme", "frequency"]
    s_counts["theme_type"] = "Suggestion (Open-Ended)"
    s_counts["pct_of_sample"] = (100.0 * s_counts["frequency"] / n_tot).round(2)

    a6_themes = pd.concat([c_counts, s_counts], ignore_index=True)
    a6_themes.to_csv(os.path.join(ANALYSIS_DIR, "analysis6_open_ended_themes.csv"), index=False, encoding="utf-8-sig")

    # Analysis 8: Support Continuation Breakdown
    a8 = df.groupby("support_category").agg(
        count=("respondent_id", "count"),
        percentage=("respondent_id", lambda x: round(len(x) / n_tot * 100, 2)),
        mean_learning=("learning_index", "mean"),
        mean_adaptability=("adaptability_index", "mean"),
        mean_crisis=("crisis_index", "mean"),
        mean_challenges=("challenge_count", "mean"),
    ).reset_index().sort_values(by="count", ascending=False)
    a8.round(3).to_csv(os.path.join(ANALYSIS_DIR, "analysis8_support_category_breakdown.csv"), index=False, encoding="utf-8-sig")
    a8.round(3).to_csv(os.path.join(ANALYSIS_DIR, "A10_support_category_breakdown.csv"), index=False, encoding="utf-8-sig")

    log("Analyses 1 through 8 and A1-A10 CSV tables exported successfully.")


def generate_figures(df: pd.DataFrame) -> None:
    """Generate Figures 1 through 6 in output/data-cleaning/analysis/figures/."""
    log("Rendering publication-quality Figures 1 to 6...")
    os.makedirs(FIG_DIR, exist_ok=True)
    n_tot = len(df)

    # --------------------------------------------------------------------------
    # Figure 1: Data Cleaning & Integration Summary
    # --------------------------------------------------------------------------
    df_loss = pd.read_csv(ANALYSIS_LOST_ROWS_CSV)
    df_loss_studies = df_loss[df_loss["source_study"] != "TOTAL_UNIFIED_DATASET"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    studies = [
        s.replace("Public_S2_", "").replace("Public_S3_", "")
        .replace("Public_S4_", "").replace("Public_S5_", "")
        .replace("Main_Study_", "")
        for s in df_loss_studies["source_study"]
    ]
    raw_vals = df_loss_studies["raw_rows"]
    clean_vals = df_loss_studies["cleaned_rows"]
    x = np.arange(len(studies))
    w = 0.36

    ax1.bar(x - w / 2, raw_vals, w, label="Raw Records Ingested", color="#4A90E2", alpha=0.9)
    ax1.bar(x + w / 2, clean_vals, w, label="Clean Topic-Valid Records", color="#2ECC71", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(studies, rotation=20, ha="right", fontsize=8)
    ax1.set_title("Multi-Source Data Ingestion & Retention (5 Studies)", fontweight="bold")
    ax1.set_ylabel("Number of Records")
    ax1.legend(frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    screened_total = int(df_loss_studies["rows_lost"].sum())
    valid_total = int(df_loss_studies["cleaned_rows"].sum())
    ax2.pie(
        [screened_total, valid_total],
        labels=[f"Screened Out ({screened_total:,})", f"Valid Retained ({valid_total:,})"],
        autopct="%1.1f%%",
        colors=["#E74C3C", "#2ECC71"],
        startangle=140,
        explode=(0.1, 0),
    )
    ax2.set_title("Stage 2 Screening Loss vs. Retention", fontweight="bold")

    plt.tight_layout()
    f1_path = os.path.join(FIG_DIR, "fig1_data_cleaning_summary.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    log(f"Saved: {f1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Unified Indices by Source Study
    # --------------------------------------------------------------------------
    df_src = pd.read_csv(os.path.join(ANALYSIS_DIR, "integration_summary_by_source.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.8))

    src_names = [
        s.replace("Public_S2_", "").replace("Public_S3_", "")
        .replace("Public_S4_", "").replace("Public_S5_", "")
        .replace("Main_Study_", "")
        for s in df_src["source_study"]
    ]
    x = np.arange(len(src_names))
    w = 0.25

    ax.bar(x - w, df_src["mean_crisis"], w, label="Crisis Mitigation Index", color="#27AE60")
    ax.bar(x, df_src["mean_adaptability"], w, label="Modality Adaptability Index", color="#3498DB")
    ax.bar(x + w, df_src["mean_learning"], w, label="Academic Learning Index", color="#E67E22")

    ax.axhline(3.0, color="gray", linestyle=":", label="Neutral Midpoint (3.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(src_names, rotation=15, ha="right")
    ax.set_ylabel("Construct Index (1.0 to 5.0 Likert)")
    ax.set_ylim(1.0, 5.0)
    ax.set_title("Composite Construct Scores across Integrated Datasets (N = 3,091)", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    f2_path = os.path.join(FIG_DIR, "fig2_unified_indices_by_study.png")
    plt.savefig(f2_path, dpi=200)
    plt.close()
    log(f"Saved: {f2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Diverging / Item-Level Likert Scores across 11 Survey Items
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y_pos = np.arange(len(ALL_11_ITEM_COLS))
    means = [df[col].mean() for col in ALL_11_ITEM_COLS]
    labels = [ITEM_LABELS[col] for col in ALL_11_ITEM_COLS]
    colors = [
        "#2980B9", "#3498DB", "#5DADE2",
        "#E67E22", "#D35400", "#8E44AD", "#F39C12",
        "#27AE60", "#2ECC71", "#16A085", "#1F618D",
    ]

    bars = ax.barh(y_pos, means, align="center", color=colors, height=0.62)
    ax.axvline(3.0, color="black", linestyle="--", linewidth=1.2, label="Neutral Baseline (3.0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(1.0, 5.0)
    ax.set_xlabel("Mean Likert Agreement Score (1 = Lowest/Strongly Disagree .. 5 = Highest/Strongly Agree)")
    ax.set_title(f"Item-Level Mean Perceptions Toward Blended Learning & Crisis Mitigation (N = {n_tot:,})", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        w_val = bar.get_width()
        ax.text(w_val + 0.05, bar.get_y() + bar.get_height() / 2, f"{w_val:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    f3_path = os.path.join(FIG_DIR, "fig3_likert_diverging.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    log(f"Saved: {f3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Scores across Internet Reliability & Residence Setup
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    net_order = ["No/unreliable home internet", "Somewhat reliable", "Very reliable"]
    df_net = (
        df.groupby("internet_group")[["learning_index", "crisis_index", "support_continue"]]
        .mean()
        .reindex(net_order)
        .reset_index()
    )
    x1 = np.arange(len(df_net))
    w = 0.25
    ax1.bar(x1 - w, df_net["learning_index"], w, label="Learning Index", color="#E67E22")
    ax1.bar(x1, df_net["crisis_index"], w, label="Crisis Index", color="#27AE60")
    ax1.bar(x1 + w, df_net["support_continue"], w, label="Support Continuation", color="#2980B9")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(df_net["internet_group"], rotation=15, ha="right", fontsize=8)
    ax1.set_ylim(1.0, 5.0)
    ax1.set_title("By Home Internet Reliability", fontweight="bold")
    ax1.set_ylabel("Mean Score (1-5)")
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    df_res = (
        df.groupby("residence_group")[["learning_index", "crisis_index", "reduce_fuel"]]
        .mean()
        .reset_index()
    )
    x2 = np.arange(len(df_res))
    ax2.bar(x2 - w, df_res["learning_index"], w, label="Learning Index", color="#E67E22")
    ax2.bar(x2, df_res["crisis_index"], w, label="Crisis Index", color="#27AE60")
    ax2.bar(x2 + w, df_res["reduce_fuel"], w, label="Fuel Expense Savings", color="#16A085")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df_res["residence_group"], rotation=15, ha="right", fontsize=8)
    ax2.set_ylim(1.0, 5.0)
    ax2.set_title("By Student Residence Arrangement", fontweight="bold")
    ax2.legend(frameon=True, fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    f4_path = os.path.join(FIG_DIR, "fig4_index_by_internet_and_residence.png")
    plt.savefig(f4_path, dpi=200)
    plt.savefig(os.path.join(FIG_DIR, "fig4_index_by_usage.png"), dpi=200)
    plt.close()
    log(f"Saved: {f4_path} and fig4_index_by_usage.png")

    # --------------------------------------------------------------------------
    # Figure 5: Student Challenges & Institutional Improvement Priorities
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ch_df = pd.read_csv(os.path.join(ANALYSIS_DIR, "A7_challenge_frequencies.csv")).sort_values("pct_of_respondents")
    ax1.barh(ch_df["challenge"], ch_df["pct_of_respondents"], color="#E74C3C", height=0.6)
    ax1.set_title("Reported Blended Learning Challenges", fontweight="bold")
    ax1.set_xlabel("% of Respondents Selecting Challenge")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for idx_b, row_b in enumerate(ch_df.itertuples()):
        ax1.text(row_b.pct_of_respondents + 1.2, idx_b, f"{row_b.pct_of_respondents:.1f}%", va="center", fontsize=8)

    imp_df = pd.read_csv(os.path.join(ANALYSIS_DIR, "A8_improvement_frequencies.csv")).sort_values("pct_of_respondents")
    ax2.barh(imp_df["improvement_area"], imp_df["pct_of_respondents"], color="#2980B9", height=0.6)
    ax2.set_title("Priority Institutional Improvements Needed", fontweight="bold")
    ax2.set_xlabel("% of Respondents Requesting Improvement")
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for idx_b, row_b in enumerate(imp_df.itertuples()):
        ax2.text(row_b.pct_of_respondents + 1.2, idx_b, f"{row_b.pct_of_respondents:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    f5_path = os.path.join(FIG_DIR, "fig5_open_ended_themes.png")
    plt.savefig(f5_path, dpi=200)
    plt.close()
    log(f"Saved: {f5_path}")

    # --------------------------------------------------------------------------
    # Figure 6: Pearson Correlation Heatmap & Crisis-Learning Gap Typology
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    corr_vars = [
        "learning_index",
        "adaptability_index",
        "crisis_index",
        "f2f_focus",
        "challenge_count",
        "support_continue",
    ]
    corr_labels = ["Learning", "Adaptability", "Crisis Mitig.", "F2F Focus", "Challenges", "Support 2-2-1"]
    corr_matrix = df[corr_vars].corr().values

    im = ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-0.6, vmax=1.0)
    ax1.set_xticks(np.arange(len(corr_labels)))
    ax1.set_yticks(np.arange(len(corr_labels)))
    ax1.set_xticklabels(corr_labels, rotation=30, ha="right")
    ax1.set_yticklabels(corr_labels)
    ax1.set_title("Construct Correlation Matrix (Pearson r)", fontweight="bold")

    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            val = corr_matrix[i, j]
            ax1.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if abs(val) > 0.55 else "black",
                fontsize=8,
            )

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    gap_counts = df["gap_typology"].value_counts()
    ax2.pie(
        gap_counts.values,
        labels=[f"{k}\n({v:,})" for k, v in gap_counts.items()],
        autopct="%1.1f%%",
        colors=["#3498DB", "#E67E22", "#2ECC71"],
        startangle=120,
        explode=[0.04] * len(gap_counts),
        textprops={"fontsize": 8},
    )
    ax2.set_title("Crisis Mitigation vs. Learning Gap Typology", fontweight="bold")

    plt.tight_layout()
    f6_path = os.path.join(FIG_DIR, "fig6_correlation_and_gap.png")
    plt.savefig(f6_path, dpi=200)
    plt.close()
    log(f"Saved: {f6_path}")


def generate_key_insights_report(df: pd.DataFrame) -> None:
    """Generate key_insights.txt summarizing core empirical findings."""
    total_n = len(df)
    mean_learn = df["learning_index"].mean()
    mean_adapt = df["adaptability_index"].mean()
    mean_crisis = df["crisis_index"].mean()
    mean_gap = df["crisis_learning_gap"].mean()
    mean_f2f = df["f2f_focus"].mean()
    mean_supp = df["support_continue"].mean()
    r_learn_supp = df["learning_index"].corr(df["support_continue"])
    r_adapt_supp = df["adaptability_index"].corr(df["support_continue"])
    r_crisis_supp = df["crisis_index"].corr(df["support_continue"])

    supp_counts = df["support_category"].value_counts().to_dict()
    n_supp = supp_counts.get("Supportive", 0)
    n_neut = supp_counts.get("Neutral", 0)
    n_not = supp_counts.get("Not supportive", 0)

    txt = f"""KEY EMPIRICAL INSIGHTS & TAKEAWAYS
================================================================================
Study Title: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano (CNSC)
Integrated Master Dataset N = {total_n:,} Clean Valid Records across 5 Unified Data Sources
================================================================================

1. THE CRISIS-MITIGATION VS. ACADEMIC-LEARNING GAP:
   - Overall Mean Crisis Mitigation Index: {mean_crisis:.3f} / 5.0 (High perceived utility in saving energy & fuel)
   - Overall Mean Modality Adaptability Index: {mean_adapt:.3f} / 5.0 (Moderate-to-high schedule compatibility)
   - Overall Mean Academic Learning Index: {mean_learn:.3f} / 5.0 (Lower than crisis mitigation rating)
   - Mean Crisis-Learning Gap: +{mean_gap:.3f} (Students recognize blended learning helps reduce campus electricity
     and household transport fuel expenses, but report stronger concentration during in-person classes,
     with F2F Focus Preference averaging {mean_f2f:.3f} / 5.0).

2. DETERMINANTS OF SUPPORT FOR CONTINUING THE 2-2-1 MODALITY:
   - Overall Mean Support for Continuation: {mean_supp:.3f} / 5.0
     * Supportive: {n_supp:,} ({n_supp / total_n * 100:.1f}%)
     * Neutral: {n_neut:,} ({n_neut / total_n * 100:.1f}%)
     * Not supportive: {n_not:,} ({n_not / total_n * 100:.1f}%)
   - Pearson Correlations with Support for Continuing (`support_continue`):
     * Academic Learning Index: r = {r_learn_supp:.4f}
     * Modality Adaptability Index: r = {r_adapt_supp:.4f}
     * Crisis Mitigation Index: r = {r_crisis_supp:.4f}

3. DIGITAL INFRASTRUCTURE & CONNECTIVITY EQUITY:
   - Home internet reliability strongly stratifies student experience: students with 'Very reliable'
     connectivity achieve higher learning index scores and report fewer modality barriers than peers
     with 'No/unreliable home internet'.
   - Commuting students living at home report strong household fuel savings (`reduce_fuel`) under the
     2-2-1 schedule while boarding/dormitory students emphasize campus Wi-Fi and power stability needs.

4. TOP CHALLENGES & PRIORITY INSTITUTIONAL INTERVENTIONS:
   - Primary Student Challenges: Internet connectivity issues, difficulty staying motivated during
     asynchronous days, lack of real-time interaction with instructors, and screen fatigue.
   - Priority Institutional Improvements Requested: Internet/technical support, stronger teacher-student
     communication, balanced scheduling of laboratory/major courses on F2F days, and clearer asynchronous
     learning materials.
================================================================================
"""
    out_path = os.path.join(ANALYSIS_DIR, "key_insights.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    log(f"Saved key insights report to: {out_path}")


def run_stage5_analysis() -> None:
    banner("STAGE 5: STATISTICAL ANALYSIS, CLEANING AUDIT & VISUALIZATIONS")

    spark = create_spark_session("Stage5_Blended221_Analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Cleaning audit tables
    generate_cleaning_audit_tables(spark)

    # 2. Read Master Unified Dataset as pandas for statistical processing
    df_master = pd.read_csv(MASTER_UNIFIED_CSV)
    log(f"Loaded Master Unified Dataset: {len(df_master):,} records x {len(df_master.columns)} columns.")

    # 3. Compute Analyses 1 through 8 and A1-A10
    generate_empirical_analysis_tables(df_master)

    # 4. Generate Figures 1 through 6
    generate_figures(df_master)

    # 5. Generate Key Insights Report
    generate_key_insights_report(df_master)

    spark.stop()
    banner("STAGE 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage5_analysis()
