"""
naive_bayes.py
==============
Gaussian Naive Bayes Classifier
Study: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano (CNSC)

Predictors (X):
  - learning_index (Academic Learning Index)
  - adaptability_index (Schedule & Transition Adaptability)
  - crisis_index (Power/Fuel Crisis Mitigation Index)
  - f2f_focus (In-Person Concentration Preference)
  - internet_num (Home Internet Reliability)
  - challenge_count (Total Modality Challenges)
Target (y): Categorical support_category (Not supportive, Neutral, Supportive)

Outputs -> `output/supervised-ml-algo/`:
  - `naive_bayes_metrics.csv`
  - `confusion_matrix.csv`
  - `figures/naive_bayes_cm.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
    print("  GAUSSIAN NAIVE BAYES CLASSIFIER — BLENDED LEARNING 2-2-1 STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS].values
    y = data["support_category"].values

    class_names = ["Not supportive", "Neutral", "Supportive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=class_names, target_names=class_names, zero_division=0))

    metrics_df = pd.DataFrame([{
        "model": "Gaussian Naive Bayes",
        "accuracy": round(accuracy, 4),
        "prior": "Empirical",
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "naive_bayes_metrics.csv"), index=False)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontweight="bold")
    ax.set_yticklabels(class_names, fontweight="bold")
    ax.set_xlabel("Predicted Modality Support Category")
    ax.set_ylabel("Actual Modality Support Category")
    ax.set_title(f"Naive Bayes Confusion Matrix (Acc = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)

    total_pts = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = (val / max(total_pts, 1)) * 100.0
            color = "white" if val > thresh else "#1E293B"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold", fontsize=8.5)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "naive_bayes_cm.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

