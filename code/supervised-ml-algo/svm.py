"""
svm.py
======
Support Vector Machine (SVM) Classifier
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
  - `svm_metrics.csv`
  - `figures/svm_decision_boundaries.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    print("  SUPPORT VECTOR MACHINE (SVM) CLASSIFIER — BLENDED LEARNING 2-2-1 STUDY")
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

    model = SVC(kernel="rbf", C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy (RBF Kernel, C=1.0): {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=class_names, target_names=class_names, zero_division=0))

    metrics_df = pd.DataFrame([{
        "model": "Support Vector Machine (SVM)",
        "kernel": "rbf",
        "C": 1.0,
        "accuracy": round(accuracy, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "svm_metrics.csv"), index=False)

    # 2D Decision Boundary visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaler.fit_transform(X))

    label_map = {"Not supportive": 0, "Neutral": 1, "Supportive": 2}
    y_num = np.array([label_map.get(lbl, 1) for lbl in y])

    svm_2d = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_2d.fit(X_pca, y_num)

    x_min, x_max = X_pca[:, 0].min() - 0.8, X_pca[:, 0].max() + 0.8
    y_min, y_max = X_pca[:, 1].min() - 0.8, X_pca[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="Blues")

    scatter_colors = ["#E74C3C", "#F39C12", "#27AE60"]
    for idx_c, lbl in enumerate(class_names):
        mask = y_num == idx_c
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=scatter_colors[idx_c],
            label=f"{lbl}",
            alpha=0.55,
            s=22,
            edgecolors="none",
        )

    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.set_title(f"SVM Decision Boundaries in Latent Space (Accuracy = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "svm_decision_boundaries.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

