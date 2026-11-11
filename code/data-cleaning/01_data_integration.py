"""
01_data_integration.py
======================
RUN ORDER: 1 of 5

Stage 1 — Multi-Source Integration for Blended Learning Pipeline
-----------------------------------------------------------------
Main Study:
  "Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis"
  Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano
  Institution: Camarines Norte State College (CNSC), Daet, Camarines Norte, Philippines

Integrates Real Datasets (Total: 1,530 raw records):
  1. Main Study (CNSC): 123 raw records -> 117 valid records (1 duplicate email, 5 straight-line screened)
  2. Public S2: Le et al. (2026), Vietnam Engineering & Multidisciplinary Universities
     (Blended learning_data set.xlsx: 580 raw records -> 508 valid, 72 straight-liners screened)
  3. Public S3: Irwanto (2024), Indonesia Undergraduate Self-Regulated Blended Learning
     (Data for SPSS.xlsx: 577 raw records -> 573 valid, 4 straight-liners screened)
  4. Public S4: Universitas Negeri Makassar (2021), Indonesia Student Perception Online Learning
     (Dataset Student Perception Online Learning.xlsx: 250 raw records -> 229 valid, 21 straight-liners screened)

Outputs -> `output/data-cleaning/integration/`:
  - `stage1_integrated_dataset.csv` (Unified dataset: 1,530 raw rows x 26 cols)
  - `stage1_lost_rows_by_study.csv` (Per-study audit of raw, valid, and screened rows)
  - `stage1_lost_rows_reasons_detail.csv` (Audit log detailing exact loss reasons)
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import openpyxl
import pandas as pd
from pyspark.sql import functions as F

from pipeline_utils import (
    INTEGRATION_DIR,
    IRWANTO_2024_XLSX,
    JUNK_TEXT_TOKENS,
    LE_2026_XLSX,
    MAIN_STUDY_CSV,
    MAKASSAR_2021_XLSX,
    STAGE1_LOST_REASONS_CSV,
    STAGE1_LOST_SUMMARY_CSV,
    STAGE1_UNIFIED_CSV,
    UNIFIED_CANONICAL_SCHEMA,
    banner,
    create_spark_session,
    log,
    write_spark_df_to_csv,
)


def map_program_group(raw_prog: str) -> str:
    """Standardize raw program/course strings into canonical program groups."""
    p = str(raw_prog or "").strip().upper()
    if not p or "MAIN CAMPUS" in p:
        return "Other / Unspecified"
    if re.search(r"BSED|BEED|BPED|COED|EDUCATION", p):
        return "Education"
    if re.search(r"INFORMATION TECHNOLOGY|BSIT|BSIY|COMPUTER", p):
        return "Information Technology"
    if re.search(r"ENGINEERING|BSCE|BSEE|BSME", p):
        return "Engineering"
    if re.search(r"AGRICULTURE|BSAG|BIO|ENVIRONMENTAL|BSES", p):
        return "Agriculture & Sciences"
    if re.search(r"BSBA|ACCOUNTANCY|^BSA\b|BPA|ENTREP|HOSPITALITY|ECONOMICS|BSOA", p):
        return "Business, Accountancy & Public Admin"
    if re.search(r"AB[ -]?ELS|ENGLISH LANGUAGE|SOCIOLOGY|BSDC", p):
        return "Arts & Social Sciences"
    return "Other / Unspecified"


def map_year_level(raw_yr: str) -> tuple[str, int]:
    """Standardize year level string and numeric year (1..4)."""
    m = re.search(r"(\d)", str(raw_yr or ""))
    yr_num = int(m.group(1)) if m else 3
    yr_num = max(1, min(4, yr_num))
    suffix = {1: "1st year", 2: "2nd year", 3: "3rd year", 4: "4th year"}[yr_num]
    return suffix, yr_num


def map_residence(raw_res: str) -> str:
    """Standardize living/residence group."""
    r = str(raw_res or "").strip().lower()
    if "dorm" in r:
        return "Dormitory/Near campus"
    if "apartment" in r or "boarding" in r:
        return "Apartment/Boarding house"
    return "At home (commuting)"


def map_internet(raw_net: str) -> tuple[str, float]:
    """Standardize home internet reliability group and numeric score (1..3)."""
    n = str(raw_net or "").strip().lower()
    if "no home internet" in n or "unreliable" in n or "no/" in n:
        return "No/unreliable home internet", 1.0
    if "somewhat" in n:
        return "Somewhat reliable", 2.0
    return "Very reliable", 3.0


def clean_text_val(val) -> str:
    """Normalize whitespace and strip placeholder junk tokens."""
    if pd.isna(val):
        return ""
    s = re.sub(r"\s+", " ", str(val)).strip()
    if s.lower() in JUNK_TEXT_TOKENS:
        return ""
    return s


def parse_likert_opt(val) -> Optional[float]:
    """Parse optional Likert value in [1.0, 5.0]; return None if missing."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        v = float(val)
        if 1.0 <= v <= 5.0:
            return round(v, 3)
    except Exception:
        pass
    return None


def text_to_likert_5(val: str) -> Optional[float]:
    """Convert text agreement strings to 1..5 Likert float."""
    v = str(val or "").strip().lower()
    if "strongly agree" in v:
        return 5.0
    if "strongly disagree" in v:
        return 1.0
    if "disagree" in v:
        return 2.0
    if "neutral" in v or "undecided" in v:
        return 3.0
    if "agree" in v:
        return 4.0
    return None


def load_main_study() -> List[Dict]:
    """Ingest and audit Primary CNSC College Students Survey (123 raw records)."""
    log("Loading Main Study (CNSC 2-2-1 Blended Learning Survey, 123 raw records)...")
    df = pd.read_csv(MAIN_STUDY_CSV, encoding="utf-8", encoding_errors="ignore")
    rows: List[Dict] = []

    likert_col_indices = [6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 21]

    for idx, r in df.iterrows():
        consent = str(r.iloc[1]).strip().lower() if pd.notna(r.iloc[1]) else "no"
        prog_grp = map_program_group(r.iloc[2])
        yr_str, yr_num = map_year_level(r.iloc[3])
        res_grp = map_residence(r.iloc[4])
        net_grp, net_num = map_internet(r.iloc[5])

        scores = [parse_likert_opt(r.iloc[i]) for i in likert_col_indices]
        non_null_scores = [s for s in scores if s is not None]
        n_missing = len(scores) - len(non_null_scores)

        exp_txt = clean_text_val(r.iloc[9])
        acad_txt = clean_text_val(r.iloc[14])
        crisis_txt = clean_text_val(r.iloc[18])
        ch_raw = str(r.iloc[19]).strip() if pd.notna(r.iloc[19]) else ""
        big_ch_txt = clean_text_val(r.iloc[20])
        imp_raw = str(r.iloc[22]).strip() if pd.notna(r.iloc[22]) else ""
        sugg_txt = clean_text_val(r.iloc[23])

        all_texts = [t for t in [exp_txt, acad_txt, crisis_txt, big_ch_txt, sugg_txt] if t]
        combined_ch_text = " | ".join([t for t in [big_ch_txt, exp_txt, acad_txt] if t])
        combined_sugg_text = " | ".join([t for t in [sugg_txt, crisis_txt] if t])

        is_valid = 1
        inv_reason = None

        # Row 118 is a documented duplicate re-submission of Row 97
        is_duplicate = (idx == 118)

        if consent != "yes":
            is_valid = 0
            inv_reason = "Screened out: Did not provide informed consent"
        elif is_duplicate:
            is_valid = 0
            inv_reason = "Screened out: Duplicate submission"
        elif n_missing > 3:
            is_valid = 0
            inv_reason = "Screened out: Incomplete response (>3 missing Likert items)"
        elif len(non_null_scores) == 11 and np.std(non_null_scores) == 0.0 and len(all_texts) == 0:
            is_valid = 0
            inv_reason = "Screened out: Straight-lining (zero variance across all 11 Likert items with blank open-ended text)"

        rows.append({
            "respondent_id": f"MAIN_{idx + 1:03d}",
            "source_study": "Main_Study_PH_CNSC_Blended_221",
            "study_role": "Main Study (PH CNSC)",
            "program_group": prog_grp,
            "year_level": yr_str,
            "year_num": yr_num,
            "residence_group": res_grp,
            "internet_group": net_grp,
            "internet_num": net_num,
            "overall_experience": scores[0],
            "schedule_fit": scores[1],
            "transition_comfort": scores[2],
            "understanding": scores[3],
            "stay_on_track": scores[4],
            "f2f_focus": scores[5],
            "academic_performance": scores[6],
            "reduce_electricity": scores[7],
            "reduce_fuel": scores[8],
            "crisis_effectiveness": scores[9],
            "support_continue": scores[10],
            "challenges_raw": ch_raw,
            "improvements_raw": imp_raw,
            "challenge_text": combined_ch_text,
            "suggestion_text": combined_sugg_text,
            "is_valid_record": is_valid,
            "invalid_reason": inv_reason,
        })

    n_val = sum(1 for r in rows if r["is_valid_record"] == 1)
    log(f"Main Study loaded: {len(rows)} raw records ({n_val} valid, {len(rows) - n_val} screened out).")
    return rows


def load_le_2026_real() -> List[Dict]:
    """
    Ingest real Le et al. (2026) Vietnam Blended Learning dataset from Excel.
    Rows: 580. Columns include:
      - CP1..CP4 (Cognitive Presence / understanding)
      - IDQ1..IDQ5 (Instructional design / schedule fit & transition)
      - LE1..LE4 (Learning experience / overall experience & manage tasks)
      - SE1..SE4 (Student engagement / staying on track)
      - AA1..AA4 (Academic achievement)
      - SP1..SP4 (Social presence / interaction)
      - GENDER, AGE, YEAR, MAJOR, PROGRAM
    """
    log("Loading Public Study 2 (Le et al., 2026 Vietnam Real Excel Data, 580 rows)...")
    wb = openpyxl.load_workbook(LE_2026_XLSX, read_only=True)
    sheet = wb.active
    rows_raw = list(sheet.iter_rows(values_only=True))
    header = rows_raw[0]
    data = rows_raw[1:]

    col_map = {name: i for i, name in enumerate(header)}
    rows: List[Dict] = []

    for idx, r in enumerate(data):
        # Major mapping
        raw_major = str(r[col_map.get("MAJOR", 35)] or "").strip()
        if "1" in raw_major or "ENG" in raw_major.upper() or "TECH" in raw_major.upper():
            prog_grp = "Engineering"
        elif "2" in raw_major or "ECON" in raw_major.upper():
            prog_grp = "Business, Accountancy & Public Admin"
        elif "3" in raw_major or "SCI" in raw_major.upper():
            prog_grp = "Agriculture & Sciences"
        else:
            prog_grp = "Engineering"

        # Year mapping
        raw_yr = r[col_map.get("YEAR", 34)]
        try:
            yr_int = int(raw_yr)
        except Exception:
            yr_int = 3
        yr_int = max(1, min(4, yr_int))
        yr_str = {1: "1st year", 2: "2nd year", 3: "3rd year", 4: "4th year"}[yr_int]

        # All Likert columns in Le are 1..5 integers
        s_exp = parse_likert_opt(r[col_map["LE1"]])
        s_sched = parse_likert_opt(r[col_map["IDQ5"]])
        s_trans = parse_likert_opt(r[col_map["IDQ2"]])
        s_under = parse_likert_opt(r[col_map["CP4"]])
        s_track = parse_likert_opt(r[col_map["SE2"]])
        s_f2f = parse_likert_opt(r[col_map["LE2"]])
        s_perf = parse_likert_opt(r[col_map["AA2"]])

        # Straight-lining check across the 30 survey Likert items (TP1 to AA4)
        likert_vals = [r[i] for i in range(2, 32) if r[i] is not None]
        is_straight = len(likert_vals) >= 25 and len(set(likert_vals)) == 1

        is_valid = 1
        inv_reason = None
        if is_straight:
            is_valid = 0
            inv_reason = "Screened out: Straight-lining (identical score across all 30 survey items)"

        rows.append({
            "respondent_id": f"LE26_{idx + 1:04d}",
            "source_study": "Public_S2_Le_2026_VN_Blended",
            "study_role": "Public Study (Le et al., 2026)",
            "program_group": prog_grp,
            "year_level": yr_str,
            "year_num": yr_int,
            "residence_group": "At home (commuting)",
            "internet_group": "Somewhat reliable",
            "internet_num": 2.0,
            "overall_experience": s_exp,
            "schedule_fit": s_sched,
            "transition_comfort": s_trans,
            "understanding": s_under,
            "stay_on_track": s_track,
            "f2f_focus": s_f2f,
            "academic_performance": s_perf,
            "reduce_electricity": None,
            "reduce_fuel": None,
            "crisis_effectiveness": None,
            "support_continue": None,
            "challenges_raw": "",
            "improvements_raw": "",
            "challenge_text": "",
            "suggestion_text": "",
            "is_valid_record": is_valid,
            "invalid_reason": inv_reason,
        })

    n_val = sum(1 for r in rows if r["is_valid_record"] == 1)
    log(f"Le et al. (2026) loaded: {len(rows)} raw records ({n_val} valid, {len(rows) - n_val} screened out).")
    return rows


def load_irwanto_2024_real() -> List[Dict]:
    """
    Ingest real Irwanto (2024) Indonesia SRL in Blended Learning from Excel.
    Rows: 577.
    Demographics: Faculty (1..8), Academic year (1..4), Location of residence (1=Rural, 2=Suburban, 3=Urban),
    Daily internet usage (1..3), Devices.
    OSLQ constructs:
      - Goal Setting (GS1..GS5)
      - Environment Structuring (ES1..ES4) -> transition_comfort
      - Task Strategies (TS1..TS4) -> stay_on_track
      - Time Management (TM1..TM3) -> schedule_fit
      - Help Seeking (HS1..HS4)
      - Self Evaluation (SE1..SE4) -> academic_performance
    """
    log("Loading Public Study 3 (Irwanto, 2024 Indonesia Real Excel Data, 577 rows)...")
    wb = openpyxl.load_workbook(IRWANTO_2024_XLSX, read_only=True)
    sheet = wb["Data for SPSS"]
    rows_raw = list(sheet.iter_rows(values_only=True))
    header = rows_raw[0]
    data = rows_raw[1:]

    col_map = {name: i for i, name in enumerate(header)}
    rows: List[Dict] = []

    faculty_name_map = {
        1: "Business, Accountancy & Public Admin",
        2: "Education",
        3: "Education",
        4: "Engineering",
        5: "Arts & Social Sciences",
        6: "Agriculture & Sciences",
        7: "Arts & Social Sciences",
        8: "Education",
    }

    residence_name_map = {
        1: "At home (commuting)",
        2: "At home (commuting)",
        3: "Dormitory/Near campus",
    }

    net_map = {
        1: ("No/unreliable home internet", 1.0),
        2: ("Somewhat reliable", 2.0),
        3: ("Very reliable", 3.0),
    }

    # Likert items
    likert_keys = [k for k in header if any(k.startswith(p) for p in ["GS", "ES", "TS", "TM", "HS", "SE"])]

    for idx, r in enumerate(data):
        fac_val = r[col_map["Faculty"]]
        prog_grp = faculty_name_map.get(fac_val, "Education")

        yr_val = r[col_map["Academic year"]]
        try:
            yr_int = int(yr_val)
        except Exception:
            yr_int = 2
        yr_int = max(1, min(4, yr_int))
        yr_str = {1: "1st year", 2: "2nd year", 3: "3rd year", 4: "4th year"}[yr_int]

        res_val = r[col_map["Location of residence"]]
        res_grp = residence_name_map.get(res_val, "At home (commuting)")

        net_val = r[col_map["Daily Internet usage"]]
        net_grp, net_num = net_map.get(net_val, ("Somewhat reliable", 2.0))

        s_sched = parse_likert_opt(r[col_map["TM1"]])
        s_trans = parse_likert_opt(r[col_map["ES1"]])
        s_under = parse_likert_opt(r[col_map["TS1"]])
        s_track = parse_likert_opt(r[col_map["GS1"]])
        s_perf = parse_likert_opt(r[col_map["SE1"]])
        s_exp = round((float(s_sched or 3.0) + float(s_under or 3.0)) / 2.0, 3)

        # Straight-lining across all 24 OSLQ items
        oslq_vals = [r[col_map[k]] for k in likert_keys if r[col_map[k]] is not None]
        is_straight = len(oslq_vals) >= 20 and len(set(oslq_vals)) == 1

        is_valid = 1
        inv_reason = None
        if is_straight:
            is_valid = 0
            inv_reason = "Screened out: Straight-lining (identical score across all 24 OSLQ items)"

        rows.append({
            "respondent_id": f"IRW24_{idx + 1:04d}",
            "source_study": "Public_S3_Irwanto_2024_ID_SRL",
            "study_role": "Public Study (Irwanto, 2024)",
            "program_group": prog_grp,
            "year_level": yr_str,
            "year_num": yr_int,
            "residence_group": res_grp,
            "internet_group": net_grp,
            "internet_num": net_num,
            "overall_experience": s_exp,
            "schedule_fit": s_sched,
            "transition_comfort": s_trans,
            "understanding": s_under,
            "stay_on_track": s_track,
            "f2f_focus": None,
            "academic_performance": s_perf,
            "reduce_electricity": None,
            "reduce_fuel": None,
            "crisis_effectiveness": None,
            "support_continue": None,
            "challenges_raw": "",
            "improvements_raw": "",
            "challenge_text": "",
            "suggestion_text": "",
            "is_valid_record": is_valid,
            "invalid_reason": inv_reason,
        })

    n_val = sum(1 for r in rows if r["is_valid_record"] == 1)
    log(f"Irwanto (2024) loaded: {len(rows)} raw records ({n_val} valid, {len(rows) - n_val} screened out).")
    return rows


def load_makassar_2021_real() -> List[Dict]:
    """
    Ingest real Universitas Negeri Makassar (2021) Student Perception Online Learning dataset from Excel.
    Rows: 250.
    Faculty: Mathematics and Natural Sciences -> Agriculture & Sciences
    Cols:
      14: Do you agree with the online learning system? -> support_continue / overall_experience
      18: Do you agree that using technology in classroom improves learning process? -> understanding
      19: Do you agree that using technology will save teachers and students time? -> schedule_fit
      20: Do you agree that using technology makes teaching and learning easy? -> transition_comfort
      22: Do you agree that technology fosters learning motivation -> stay_on_track
      24: Do you agree that technology increases creativity? -> academic_performance
    """
    log("Loading Public Study 4 (Makassar, 2021 Indonesia Real Excel Data, 250 rows)...")
    wb = openpyxl.load_workbook(MAKASSAR_2021_XLSX, read_only=True)
    sheet = wb.active
    rows_raw = list(sheet.iter_rows(values_only=True))
    data = rows_raw[1:]
    rows: List[Dict] = []

    for idx, r in enumerate(data):
        # Semester mapping (e.g. 'Semester 5/Semester 6' -> 3rd year)
        sem_str = str(r[6] or "").lower()
        if "1" in sem_str or "2" in sem_str:
            yr_str, yr_num = "1st year", 1
        elif "3" in sem_str or "4" in sem_str:
            yr_str, yr_num = "2nd year", 2
        elif "5" in sem_str or "6" in sem_str:
            yr_str, yr_num = "3rd year", 3
        else:
            yr_str, yr_num = "4th year", 4

        s_supp = text_to_likert_5(r[14])
        s_under = text_to_likert_5(r[18])
        s_sched = text_to_likert_5(r[19])
        s_trans = text_to_likert_5(r[20])
        s_track = text_to_likert_5(r[22])
        s_perf = text_to_likert_5(r[24])

        # Straight-lining across all 16 agreement items (cols 12 to 27)
        agree_vals = [text_to_likert_5(r[i]) for i in range(12, 28) if text_to_likert_5(r[i]) is not None]
        is_straight = len(agree_vals) >= 14 and len(set(agree_vals)) == 1

        is_valid = 1
        inv_reason = None
        if is_straight:
            is_valid = 0
            inv_reason = "Screened out: Straight-lining (identical agreement rating across all 16 survey items)"

        rows.append({
            "respondent_id": f"MAK21_{idx + 1:04d}",
            "source_study": "Public_S4_Makassar_2021_Student_Perception",
            "study_role": "Public Study (Makassar, 2021)",
            "program_group": "Agriculture & Sciences",
            "year_level": yr_str,
            "year_num": yr_num,
            "residence_group": "At home (commuting)",
            "internet_group": "Somewhat reliable",
            "internet_num": 2.0,
            "overall_experience": s_supp,
            "schedule_fit": s_sched,
            "transition_comfort": s_trans,
            "understanding": s_under,
            "stay_on_track": s_track,
            "f2f_focus": None,
            "academic_performance": s_perf,
            "reduce_electricity": None,
            "reduce_fuel": None,
            "crisis_effectiveness": None,
            "support_continue": s_supp,
            "challenges_raw": "",
            "improvements_raw": "",
            "challenge_text": "",
            "suggestion_text": "",
            "is_valid_record": is_valid,
            "invalid_reason": inv_reason,
        })

    n_val = sum(1 for r in rows if r["is_valid_record"] == 1)
    log(f"Makassar (2021) loaded: {len(rows)} raw records ({n_val} valid, {len(rows) - n_val} screened out).")
    return rows


def run_stage1_integration() -> None:
    banner("STAGE 1: TOPIC-ALIGNED MULTI-SOURCE DATA INTEGRATION (REAL DATASETS)")

    spark = create_spark_session("Stage1_Blended_Real_Integration")
    os.makedirs(INTEGRATION_DIR, exist_ok=True)

    all_rows: List[Dict] = []
    all_rows.extend(load_main_study())
    all_rows.extend(load_le_2026_real())
    all_rows.extend(load_irwanto_2024_real())
    all_rows.extend(load_makassar_2021_real())

    total_raw = len(all_rows)
    total_valid = sum(1 for r in all_rows if r["is_valid_record"] == 1)
    total_screened = total_raw - total_valid

    log(f"Combined Integration Count: {total_raw:,} total raw rows across 4 studies.")
    log(f"Valid Topic-Related Rows: {total_valid:,} | Screened/Invalid Rows: {total_screened:,}")

    pdf = pd.DataFrame(all_rows)
    df_spark = spark.createDataFrame(pdf)
    df_unified = df_spark.select(UNIFIED_CANONICAL_SCHEMA)

    log(f"Writing Stage 1 Unified CSV to: {STAGE1_UNIFIED_CSV}")
    written = write_spark_df_to_csv(df_unified, STAGE1_UNIFIED_CSV)
    log(f"Successfully saved {written:,} rows to Stage 1 Unified Dataset.")

    df_audit = df_unified.groupBy("source_study", "study_role").agg(
        F.count("*").alias("raw_count"),
        F.sum(F.when(F.col("is_valid_record") == 1, 1).otherwise(0)).alias("valid_count"),
        F.sum(F.when(F.col("is_valid_record") == 0, 1).otherwise(0)).alias("screened_lost_count"),
    ).orderBy("raw_count")

    write_spark_df_to_csv(df_audit, STAGE1_LOST_SUMMARY_CSV)
    log(f"Saved Stage 1 Lost Summary Audit to: {STAGE1_LOST_SUMMARY_CSV}")

    df_reasons = (
        df_unified.filter(F.col("is_valid_record") == 0)
        .groupBy("source_study", "invalid_reason")
        .count()
        .withColumnRenamed("count", "lost_rows")
        .orderBy("source_study", F.desc("lost_rows"))
    )
    write_spark_df_to_csv(df_reasons, STAGE1_LOST_REASONS_CSV)
    log(f"Saved Stage 1 Loss Reasons Audit to: {STAGE1_LOST_REASONS_CSV}")

    spark.stop()
    banner("STAGE 1 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage1_integration()
