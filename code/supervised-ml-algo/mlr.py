"""
mlr.py
======
Multiple Linear Regression (MLR)
Study: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano (CNSC)

Predictors (X):
  - learning_index (Understanding, Staying on Track, Academic Performance)
  - adaptability_index (Schedule Fit & Modality Transition Comfort)
  - crisis_index (Electricity Savings, Fuel Savings, Crisis Effectiveness)
  - f2f_focus (Preference for In-Person Concentration)
  - internet_num (Home Internet Reliability Score)
  - challenge_count (Total Modality Challenges Experienced)
Target (y): Continuous support_continue [1.0 to 5.0]

Outputs -> `output/supervised-ml-algo/`:
  - `mlr_metrics.csv`
  - `mlr_coefficients.csv`
  - `figures/mlr_fit.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "supervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

FEATURE_COLS = [
    "learning_index",
    "adaptability_index",
    "crisis_index",
    "f2f_focus",
    "internet_num",
    "challenge_count",
]
FEATURE_LABELS = {
    "learning_index": "Academic Learning Index",
    "adaptability_index": "Schedule & Transition Adaptability",
    "crisis_index": "Power/Fuel Crisis Mitigation Index",
    "f2f_focus": "F2F Focus Preference",
    "internet_num": "Home Internet Reliability",
    "challenge_count": "Total Modality Challenges",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "sans-serif",
})


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  MULTIPLE LINEAR REGRESSION (MLR) — BLENDED LEARNING 2-2-1 STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        if c not in data.columns:
            data[c] = 0.0
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS]
    y = data["support_continue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"RMSE              : {rmse:.4f}")
    print(f"MAE               : {mae:.4f}")
    print(f"R-squared (R2)    : {r2:.4f}")
    print("\nModel Coefficients:")
    coef_rows = []
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        lbl = FEATURE_LABELS.get(feat, feat)
        print(f"  {lbl:<36}: {coef:+.4f}")
        coef_rows.append({"feature": feat, "display_name": lbl, "coefficient": round(coef, 4)})
    coef_rows.append({"feature": "intercept", "display_name": "Model Intercept", "coefficient": round(model.intercept_, 4)})
    print(f"  {'Model Intercept':<36}: {model.intercept_:+.4f}")

    pd.DataFrame(coef_rows).to_csv(os.path.join(OUTPUT_DIR, "mlr_coefficients.csv"), index=False)
    pd.DataFrame(coef_rows).to_csv(os.path.join(OUTPUT_DIR, "regression_coefficients.csv"), index=False)

    metrics_df = pd.DataFrame([{
        "model": "Multiple Linear Regression (MLR)",
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n_predictors": len(FEATURE_COLS),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "mlr_metrics.csv"), index=False)

    # Visualization: Predicted vs. Actual
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(y_test, y_pred, color="#1ABC9C", alpha=0.45, s=22, label="Test Set Observations")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#E67E22", linestyle="--", linewidth=2.0, label="Ideal Prediction (45° Line)")

    ax.text(
        0.05, 0.85,
        f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nPredictors = {len(FEATURE_COLS)}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    ax.set_xlabel("Actual Support for Continuing 2-2-1 Score")
    ax.set_ylabel("Predicted Support Score (MLR)")
    ax.set_title("Multiple Linear Regression: Actual vs. Predicted Modality Support", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "mlr_fit.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

