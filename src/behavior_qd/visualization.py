"""Visualization utilities for behavior-qd framework."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from ribs.visualize import cvt_archive_heatmap

from behavior_qd.archive import BehaviorArchive


def plot_archive_2d(
    archive: BehaviorArchive,
    dims: tuple[int, int] = (0, 1),
    title: str = "Archive Heatmap",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
) -> Figure | None:
    """Plot a 2D heatmap of the archive.

    Projects the 3D archive onto 2 specified dimensions.
    Note: cvt_archive_heatmap only works for 1D or 2D archives.
    For 3D archives, this returns None.

    Args:
        archive: The behavior archive to visualize.
        dims: Which dimensions to plot (0=PCA1, 1=PCA2, 2=variance).
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        Matplotlib Figure object, or None if archive is 3D.
    """
    # Check archive dimensionality - cvt_archive_heatmap only works for 1D/2D
    # In pyribs 0.9+, use measure_dim attribute
    measure_dim = archive.archive.measure_dim
    if measure_dim > 2:
        # Can't use cvt_archive_heatmap for 3D, return None
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Use pyribs CVT heatmap
    cvt_archive_heatmap(
        archive.archive,
        ax=ax,
        cmap=cmap,
    )

    dim_names = ["PCA Dim 1", "PCA Dim 2", "Token Variance"]
    ax.set_xlabel(dim_names[dims[0]])
    ax.set_ylabel(dim_names[dims[1]])
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_archive_3d(
    archive: BehaviorArchive,
    title: str = "Archive 3D View",
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = 45,
) -> Figure:
    """Plot a 3D scatter of the archive.

    Args:
        archive: The behavior archive to visualize.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.
        elev: Elevation angle for 3D view.
        azim: Azimuth angle for 3D view.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get elite data
    elites = archive.get_elites()

    if not elites:
        ax.set_title(f"{title} (Empty)")
        return fig

    # Extract measures and objectives
    pca1 = [e.measures.pca_1 for e in elites]
    pca2 = [e.measures.pca_2 for e in elites]
    variance = [e.measures.variance for e in elites]
    objectives = [e.objective for e in elites]

    # Create scatter plot
    scatter = ax.scatter(
        pca1,
        pca2,
        variance,
        c=objectives,
        cmap=cmap,
        s=50,
        alpha=0.7,
    )

    ax.set_xlabel("PCA Dim 1")
    ax.set_ylabel("PCA Dim 2")
    ax.set_zlabel("Token Variance")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    plt.colorbar(scatter, ax=ax, label="Objective Score")
    plt.tight_layout()

    return fig


def plot_objective_history(
    stats_iterations: list[dict],
    title: str = "Objective Over Iterations",
    figsize: tuple[int, int] = (12, 5),
) -> Figure:
    """Plot objective score history over iterations.

    Args:
        stats_iterations: List of iteration stats dicts.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    iterations = [s["iteration"] for s in stats_iterations]
    best_scores = [s["best_score"] for s in stats_iterations]
    mean_scores = [s["mean_score"] for s in stats_iterations]
    qd_scores = [s["qd_score"] for s in stats_iterations]
    archive_sizes = [s["archive_size"] for s in stats_iterations]

    # Best and mean objective
    axes[0].plot(iterations, best_scores, label="Best", color="green")
    axes[0].plot(iterations, mean_scores, label="Mean", color="blue", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective Score")
    axes[0].set_title("Objective Scores")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # QD Score
    axes[1].plot(iterations, qd_scores, color="purple")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("QD Score")
    axes[1].set_title("QD Score")
    axes[1].grid(True, alpha=0.3)

    # Archive size
    axes[2].plot(iterations, archive_sizes, color="orange")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Archive Size")
    axes[2].set_title("Archive Size")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def export_best_prompts(
    archive: BehaviorArchive,
    output_path: Path | str,
    n: int = 100,
    format: str = "json",
) -> None:
    """Export the best prompts from the archive.

    Args:
        archive: The behavior archive.
        output_path: Path to save the export.
        n: Number of top prompts to export.
        format: Output format ("json", "csv", or "txt").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    elites = archive.get_elites(n)

    if format == "json":
        import json

        data = [
            {
                "prompt": e.prompt,
                "objective": e.objective,
                "pca_1": e.measures.pca_1,
                "pca_2": e.measures.pca_2,
                "variance": e.measures.variance,
                "response": e.response,
                "reasoning": e.reasoning,
            }
            for e in elites
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "prompt", "objective", "pca_1", "pca_2", "variance", "response", "reasoning"
            ])
            for e in elites:
                writer.writerow([
                    e.prompt,
                    e.objective,
                    e.measures.pca_1,
                    e.measures.pca_2,
                    e.measures.variance,
                    e.response or "",
                    e.reasoning or "",
                ])

    elif format == "txt":
        with open(output_path, "w") as f:
            for i, e in enumerate(elites, 1):
                f.write(f"=== Prompt {i} (Score: {e.objective:.3f}) ===\n")
                f.write(f"{e.prompt}\n")
                if e.response:
                    f.write(f"\nResponse: {e.response[:200]}...\n")
                if e.reasoning:
                    f.write(f"\nReasoning: {e.reasoning}\n")
                f.write("\n")

    else:
        raise ValueError(f"Unknown format: {format}")


def save_all_visualizations(
    archive: BehaviorArchive,
    output_dir: Path | str,
    stats_iterations: list[dict] | None = None,
) -> None:
    """Save all visualizations to a directory.

    Args:
        archive: The behavior archive.
        output_dir: Directory to save visualizations.
        stats_iterations: Optional iteration stats for history plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2D heatmap (only works for 1D/2D archives)
    fig = plot_archive_2d(archive)
    if fig is not None:
        fig.savefig(output_dir / "archive_heatmap_2d.png", dpi=150)
        plt.close(fig)

    # 3D scatter (works for any archive dimensionality)
    fig = plot_archive_3d(archive)
    fig.savefig(output_dir / "archive_scatter_3d.png", dpi=150)
    plt.close(fig)

    # History plots if available
    if stats_iterations:
        fig = plot_objective_history(stats_iterations)
        fig.savefig(output_dir / "objective_history.png", dpi=150)
        plt.close(fig)

    # Export prompts in multiple formats
    export_best_prompts(archive, output_dir / "best_prompts.json", n=100, format="json")
    export_best_prompts(archive, output_dir / "best_prompts.csv", n=100, format="csv")
    export_best_prompts(archive, output_dir / "best_prompts.txt", n=20, format="txt")
