#!/usr/bin/env python3
"""Debug script to understand why prompts are being rejected from the archive.

This script:
1. Loads the embedding space and creates a test archive
2. Computes measures for seed prompts (and optionally generated prompts)
3. Visualizes the measure space to understand clustering
4. Simulates adding prompts to identify rejection reasons

Usage:
    uv run python scripts/debug_archive.py
    uv run python scripts/debug_archive.py --prompts-file output/latest_prompts.txt
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from behavior_qd.archive import BehaviorArchive
from behavior_qd.config import ArchiveConfig, EmbeddingConfig
from behavior_qd.embeddings import EmbeddingSpace, Measures

console = Console()


def load_seed_prompts(seed_file: Path = Path("seed_prompts.csv")) -> list[str]:
    """Load prompts from seed CSV file."""
    prompts = []
    with open(seed_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row["prompt"].strip().strip('"')
            if prompt:
                prompts.append(prompt)
    return prompts


def load_prompts_from_file(filepath: Path) -> list[str]:
    """Load prompts from a text file (one per line)."""
    prompts = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def compute_all_measures(
    prompts: list[str], embedding_space: EmbeddingSpace
) -> list[tuple[str, Measures]]:
    """Compute measures for all prompts."""
    results = []
    for prompt in prompts:
        measures = embedding_space.compute_measures(prompt)
        results.append((prompt, measures))
    return results


def analyze_measure_distribution(
    prompt_measures: list[tuple[str, Measures]],
    archive_config: ArchiveConfig,
) -> dict:
    """Analyze the distribution of measures."""
    dim1_vals = [m.dim1 for _, m in prompt_measures]
    dim2_vals = [m.dim2 for _, m in prompt_measures]

    dim_range = archive_config.dim_range

    stats = {
        "dim1": {
            "min": min(dim1_vals),
            "max": max(dim1_vals),
            "mean": np.mean(dim1_vals),
            "std": np.std(dim1_vals),
            "range": dim_range,
        },
        "dim2": {
            "min": min(dim2_vals),
            "max": max(dim2_vals),
            "mean": np.mean(dim2_vals),
            "std": np.std(dim2_vals),
            "range": dim_range,
        },
    }

    # Check how many are out of range
    out_of_range = 0
    for _, m in prompt_measures:
        if not (dim_range[0] <= m.dim1 <= dim_range[1]):
            out_of_range += 1
        elif not (dim_range[0] <= m.dim2 <= dim_range[1]):
            out_of_range += 1

    stats["out_of_range_count"] = out_of_range
    stats["out_of_range_pct"] = out_of_range / len(prompt_measures) * 100

    return stats


def simulate_archive_additions(
    prompt_measures: list[tuple[str, Measures]],
    scores: list[float],
    archive: BehaviorArchive,
) -> dict:
    """Simulate adding prompts to the archive and track results."""
    results = {
        "added": [],
        "improved": [],
        "rejected": [],
        "cell_collisions": defaultdict(list),  # cell_id -> list of (prompt, score)
    }

    for (prompt, measures), score in zip(prompt_measures, scores):
        # Get the cell this would map to
        cell_idx = archive.archive.index_of_single(measures.as_array())

        # Track collision
        results["cell_collisions"][cell_idx].append((prompt[:80], score, measures))

        # Try to add
        status, added_cell = archive.add(
            prompt=prompt,
            objective=score,
            measures=measures,
        )

        status_int = int(status)
        if status_int == 0:
            results["rejected"].append((prompt, score, measures, cell_idx))
        elif status_int == 1:
            results["added"].append((prompt, score, measures, cell_idx))
        elif status_int == 2:
            results["improved"].append((prompt, score, measures, cell_idx))

    return results


def print_measure_stats(stats: dict):
    """Print measure distribution statistics."""
    table = Table(title="Measure Distribution Statistics")
    table.add_column("Measure", style="cyan")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Archive Range", justify="right")
    table.add_column("In Range?", justify="center")

    for measure_name in ["dim1", "dim2"]:
        s = stats[measure_name]
        in_range = s["range"][0] <= s["min"] and s["max"] <= s["range"][1]
        table.add_row(
            measure_name,
            f"{s['min']:.4f}",
            f"{s['max']:.4f}",
            f"{s['mean']:.4f}",
            f"{s['std']:.4f}",
            f"[{s['range'][0]}, {s['range'][1]}]",
            "[green]Yes[/green]" if in_range else "[red]No[/red]",
        )

    console.print(table)
    console.print(
        f"\n[yellow]Prompts with measures out of archive range: "
        f"{stats['out_of_range_count']} ({stats['out_of_range_pct']:.1f}%)[/yellow]"
    )


def print_simulation_results(results: dict):
    """Print simulation results."""
    console.print("\n[bold]Archive Simulation Results[/bold]")
    console.print(f"  Added (new cells): {len(results['added'])}")
    console.print(f"  Improved (better score): {len(results['improved'])}")
    console.print(f"  Rejected: {len(results['rejected'])}")

    # Find cells with collisions
    collisions = {k: v for k, v in results["cell_collisions"].items() if len(v) > 1}
    if collisions:
        console.print(f"\n[yellow]Cell Collisions ({len(collisions)} cells):[/yellow]")
        for cell_idx, entries in sorted(collisions.items(), key=lambda x: -len(x[1]))[
            :10
        ]:
            console.print(f"\n  Cell {cell_idx} ({len(entries)} prompts):")
            for prompt_snippet, score, measures in entries:
                console.print(
                    f"    - [{score:.2f}] {prompt_snippet}..."
                    f"\n      measures: ({measures.dim1:.3f}, {measures.dim2:.3f})"
                )


def plot_measure_space(
    prompt_measures: list[tuple[str, Measures]],
    archive_config: ArchiveConfig,
    output_path: Path = Path("output/debug_measures.png"),
):
    """Create visualization of the measure space."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dim1 = [m.dim1 for _, m in prompt_measures]
    dim2 = [m.dim2 for _, m in prompt_measures]

    # Scatter of dim1 vs dim2
    ax = axes[0]
    ax.scatter(dim1, dim2, alpha=0.6, s=100)
    for i, (prompt, m) in enumerate(prompt_measures):
        ax.annotate(
            str(i),
            (m.dim1, m.dim2),
            fontsize=8,
            ha="center",
            va="bottom",
        )
    # Draw archive range
    dim_range = archive_config.dim_range
    ax.axhline(y=dim_range[0], color="r", linestyle="--", alpha=0.5)
    ax.axhline(y=dim_range[1], color="r", linestyle="--", alpha=0.5)
    ax.axvline(x=dim_range[0], color="r", linestyle="--", alpha=0.5)
    ax.axvline(x=dim_range[1], color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title("Sentence Embedding Projection (red = archive bounds)")

    # Histogram of pairwise distances between prompts
    ax = axes[1]
    from scipy.spatial.distance import pdist

    all_measures = np.array(
        [[m.dim1, m.dim2] for _, m in prompt_measures]
    )
    if len(all_measures) > 1:
        distances = pdist(all_measures)
        ax.hist(distances, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            x=np.mean(distances),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(distances):.3f}",
        )
    ax.set_xlabel("Pairwise Distance")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Distances in Measure Space")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    console.print(f"\n[green]Saved visualization to {output_path}[/green]")


def print_prompt_details(prompt_measures: list[tuple[str, Measures]]):
    """Print detailed info for each prompt."""
    table = Table(title="Prompt Measures")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Dim 1", justify="right")
    table.add_column("Dim 2", justify="right")
    table.add_column("Prompt (truncated)", max_width=60)

    for i, (prompt, m) in enumerate(prompt_measures):
        table.add_row(
            str(i),
            f"{m.dim1:.4f}",
            f"{m.dim2:.4f}",
            prompt[:60] + "..." if len(prompt) > 60 else prompt,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Debug archive behavior")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Additional prompts file to analyze (one per line)",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=Path("seed_prompts.csv"),
        help="Seed prompts CSV file",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=1000,
        help="Number of archive cells",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    console.print("[bold blue]Archive Debug Script[/bold blue]\n")

    # Load prompts
    console.print("[dim]Loading prompts...[/dim]")
    prompts = load_seed_prompts(args.seed_file)
    console.print(f"Loaded {len(prompts)} seed prompts")

    if args.prompts_file and args.prompts_file.exists():
        extra_prompts = load_prompts_from_file(args.prompts_file)
        console.print(
            f"Loaded {len(extra_prompts)} additional prompts from {args.prompts_file}"
        )
        prompts.extend(extra_prompts)

    # Initialize embedding space
    console.print("\n[dim]Loading embedding model...[/dim]")
    embedding_config = EmbeddingConfig()
    embedding_space = EmbeddingSpace(embedding_config)
    # Trigger lazy loading
    _ = embedding_space.pca
    console.print(
        f"[green]Loaded {embedding_config.model_name}[/green] "
        f"(dim={embedding_space.embed_dim})"
    )

    # Compute measures
    console.print("\n[dim]Computing measures for all prompts...[/dim]")
    prompt_measures = compute_all_measures(prompts, embedding_space)

    # Print prompt details
    print_prompt_details(prompt_measures)

    # Analyze distribution
    archive_config = ArchiveConfig(num_cells=args.num_cells)
    stats = analyze_measure_distribution(prompt_measures, archive_config)
    print_measure_stats(stats)

    # Create archive and simulate additions
    console.print("\n[dim]Creating test archive and simulating additions...[/dim]")
    archive = BehaviorArchive(archive_config, embedding_space)

    # Use fake scores (varying to see competition effects)
    # Higher scores for first prompts to see if later ones get rejected
    fake_scores = [7.0 - i * 0.5 for i in range(len(prompts))]
    console.print(f"Using fake scores: {fake_scores[:5]}... (descending)")

    results = simulate_archive_additions(prompt_measures, fake_scores, archive)
    print_simulation_results(results)

    # Now try with all same scores to see pure geometric clustering
    console.print("\n[bold]Re-testing with equal scores (5.0 for all)...[/bold]")
    archive2 = BehaviorArchive(archive_config, embedding_space)
    equal_scores = [5.0] * len(prompts)
    results2 = simulate_archive_additions(prompt_measures, equal_scores, archive2)
    print_simulation_results(results2)

    # Plot
    if not args.no_plot:
        console.print("\n[dim]Generating visualization...[/dim]")
        plot_measure_space(prompt_measures, archive_config)

    # Summary
    console.print("\n[bold green]Summary[/bold green]")
    console.print(
        "If prompts cluster in the same cells, they're too similar in measure space."
    )
    console.print(
        "If measures are outside archive range, they'll be clipped and may collide."
    )
    console.print("\nPotential fixes:")
    console.print("  1. Expand archive ranges to match actual measure distribution")
    console.print("  2. Use more cells (currently {})".format(args.num_cells))
    console.print("  3. Change embedding model or PCA components")
    console.print("  4. Add diversity pressure in prompt generation")


if __name__ == "__main__":
    main()
