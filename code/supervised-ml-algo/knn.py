"""
knn.py
======
K-Nearest Neighbors (KNN) Classifier
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
  - `knn_metrics.csv`
  - `figures/knn_accuracy_k.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
    print("  K-NEAREST NEIGHBORS (KNN) CLASSIFIER — BLENDED LEARNING 2-2-1 STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS].values
    y = data["support_category"].values

    class_names = ["Not supportive", "Neutral", "Supportive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_chosen = 5
    model = KNeighborsClassifier(n_neighbors=k_chosen)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc_chosen = accuracy_score(y_test, y_pred)

    print(f"Accuracy (k = {k_chosen}): {acc_chosen * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=class_names, target_names=class_names, zero_division=0))

    # Evaluate accuracy across k = 1..15
    k_range = range(1, 16)
    accuracies = []
    for k_val in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k_val)
        knn_k.fit(X_train_scaled, y_train)
        pred_k = knn_k.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, pred_k))

    best_k = list(k_range)[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print(f"\nBest k: {best_k} with Accuracy: {best_acc * 100:.2f}%")

    metrics_df = pd.DataFrame([{
        "model": "K-Nearest Neighbors (KNN)",
        "chosen_k": k_chosen,
        "chosen_k_accuracy": round(acc_chosen, 4),
        "optimal_k": best_k,
        "optimal_k_accuracy": round(best_acc, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "knn_metrics.csv"), index=False)

    # Plot Accuracy vs. k
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k_range, [a * 100 for a in accuracies], marker="o", color="#2980B9", linewidth=2.0)
    ax.axvline(best_k, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Best k = {best_k} ({best_acc * 100:.1f}%)")
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_xticks(list(k_range))
    ax.set_title("KNN Classification Accuracy Across k Values (Modality Support)", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "knn_accuracy_k.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

