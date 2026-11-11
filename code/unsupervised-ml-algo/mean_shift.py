"""
mean_shift.py
=============
Unsupervised Machine Learning — Mean-Shift Density Clustering
Study: Perceptions of College Students Toward Blended Learning as a Response to Power Fuel Crisis
Authors: Johann Jethro O. Chavez, Mikylla Nicole M. Nañez, Divine Mae S. Ponciano (CNSC)

Performs:
  1. Standardizes core construct features (learning_index, adaptability_index, crisis_index,
     f2f_focus, internet_num, challenge_count).
  2. Estimates optimal kernel bandwidth automatically using `estimate_bandwidth(quantile=0.20)`.
  3. Fits Mean-Shift (density-based mode-seeking clustering) to automatically discover natural modes without presetting k.
  4. Profiles discovered density modes with construct means and population shares.
  5. Projects density clusters into 2D PCA space with marked mode centers.

Outputs -> `output/unsupervised-ml-algo/`:
  - `mean_shift_metrics.csv`
  - `mean_shift_profiles.csv`
  - `figures/fig4_mean_shift_clusters.png`
  - `figures/fig5_mean_shift_profiles.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "unsupervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

CONSTRUCT_COLS = [
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


def plot_fig4_mean_shift_clusters(X_pca: np.ndarray, labels: np.ndarray, centers_pca: np.ndarray, n_clusters: int) -> None:
    """Figure 4: Mean-Shift Clusters Projected into 2D PCA Space with Centroids."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("viridis", max(n_clusters, 2))

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.45, s=20, edgecolors="none")

    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        s=160,
        c="red",
        marker="X",
        edgecolors="black",
        linewidth=1.5,
        label="Discovered Density Modes (Centers)",
        zorder=6,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean-Shift Cluster Label")
    ax.set_xlabel("PCA Component 1 (Overall Blended Learning & Adaptability Axis)")
    ax.set_ylabel("PCA Component 2 (Crisis Conservation vs. F2F Preference Axis)")
    ax.set_title(f"Figure 4: Mean-Shift Density Clustering in PCA Space ({n_clusters} Modes)", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper right", fontsize=8.2)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_mean_shift_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig5_mean_shift_profiles(profile_df: pd.DataFrame) -> None:
    """Figure 5: Mean Construct Scores across Discovered Mean-Shift Density Clusters."""
    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    x = np.arange(len(profile_df))
    w = 0.26

    c1 = ax.bar(x - w, profile_df["mean_learning"], w, label="Learning Index", color="#E67E22", edgecolor="white")
    c2 = ax.bar(x, profile_df["mean_adaptability"], w, label="Adaptability Index", color="#3498DB", edgecolor="white")
    c3 = ax.bar(x + w, profile_df["mean_crisis"], w, label="Crisis Mitigation Index", color="#2ECC71", edgecolor="white")

    ax.bar_label(c1, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c2, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c3, fmt="%.2f", padding=2, fontsize=7.8)

    ax.set_xticks(x)
    x_labels = [
        f"Mode {int(r['cluster_id'])}\n({r['share_pct']:.1f}%)\nn={int(r['count']):,}"
        for _, r in profile_df.iterrows()
    ]
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.2)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Construct Score (1.0 to 5.0 Likert)")
    ax.set_title(f"Figure 5: Construct Means Across Mean-Shift Modes (N = {int(profile_df['count'].sum()):,})", fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_mean_shift_profiles.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  MEAN-SHIFT CLUSTERING (NON-PARAMETRIC MODES) — BLENDED LEARNING 2-2-1 STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in CONSTRUCT_COLS:
        data[c] = data[c].fillna(data[c].median())

    X = data[CONSTRUCT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Estimate Bandwidth
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.18, n_samples=1000, random_state=42)
    print(f"Estimated Kernel Bandwidth (quantile=0.18): {bandwidth:.4f}")

    # 2. Fit Mean-Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X_scaled)
    data["mean_shift_cluster"] = labels

    cluster_centers = ms.cluster_centers_
    n_clusters = len(np.unique(labels))
    print(f"Number of Discovered Natural Modes (Clusters): {n_clusters}")

    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0.0
    print(f"Mean-Shift Silhouette Score: {sil:.4f}")

    # Export metrics
    metrics_df = pd.DataFrame([{
        "model": "Mean-Shift Clustering",
        "estimated_bandwidth": round(bandwidth, 4),
        "clusters_discovered": n_clusters,
        "silhouette_score": round(sil, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "mean_shift_metrics.csv"), index=False)

    # 3. Profile Clusters
    total_n = len(data)
    prof_rows = []
    for c_id in sorted(np.unique(labels)):
        subset = data[data["mean_shift_cluster"] == c_id]
        prof_rows.append({
            "cluster_id": c_id,
            "count": len(subset),
            "share_pct": round(len(subset) / total_n * 100, 2),
            "mean_learning": round(subset["learning_index"].mean(), 3),
            "mean_adaptability": round(subset["adaptability_index"].mean(), 3),
            "mean_crisis": round(subset["crisis_index"].mean(), 3),
            "mean_f2f_focus": round(subset["f2f_focus"].mean(), 3),
            "mean_support": round(subset["support_continue"].mean(), 3),
        })

    profile_df = pd.DataFrame(prof_rows).sort_values("count", ascending=False).reset_index(drop=True)
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "mean_shift_profiles.csv"), index=False)
    print("\nMean-Shift Discovered Mode Profiles:")
    print(profile_df.to_string(index=False))

    # 4. PCA Projection & Figures
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(cluster_centers)

    plot_fig4_mean_shift_clusters(X_pca, labels, centers_pca, n_clusters)
    plot_fig5_mean_shift_profiles(profile_df)
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

