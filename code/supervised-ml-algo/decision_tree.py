"""
decision_tree.py
================
Decision Tree Classifier
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
  - `decision_tree_metrics.csv`
  - `feature_importances.csv`
  - `figures/decision_tree.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
FEATURE_NAMES = [
    "Learning Index",
    "Adaptability Index",
    "Crisis Index",
    "F2F Focus",
    "Internet Reliability",
    "Challenge Count",
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
    print("  DECISION TREE CLASSIFIER — BLENDED LEARNING 2-2-1 STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS].values
    y = data["support_category"].values

    class_names = ["Not supportive", "Neutral", "Supportive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=class_names, target_names=class_names, zero_division=0))

    print("\nFeature Importances:")
    imp_rows = []
    for feat, imp in zip(FEATURE_NAMES, model.feature_importances_):
        print(f"  {feat:<24}: {imp * 100:.2f}%")
        imp_rows.append({"feature": feat, "importance": round(imp, 4), "importance_pct": round(imp * 100, 2)})

    pd.DataFrame(imp_rows).to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

    metrics_df = pd.DataFrame([{
        "model": "Decision Tree Classifier",
        "accuracy": round(accuracy, 4),
        "max_depth": 3,
        "n_classes": len(class_names),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_metrics.csv"), index=False)

    # Visualize tree
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    tree.plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision Tree for 2-2-1 Modality Support Category (Accuracy = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "decision_tree.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

