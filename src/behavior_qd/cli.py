"""Command-line interface for behavior-qd framework."""

import json
from pathlib import Path
from typing import Optional

import typer
from flashlite import DiskCache
from rich.console import Console
from rich.table import Table

from behavior_qd.config import BehaviorQDConfig, EmitterMode

app = typer.Typer(
    name="behavior-qd",
    help="Zero-shot behavior elicitation using Quality Diversity optimization.",
)
console = Console()


@app.command()
def run(
    behavior: str = typer.Argument(
        ...,
        help="Description of the behavior to elicit (e.g., 'sycophancy')",
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Directory for outputs and checkpoints",
    ),
    mode: str = typer.Option(
        "sampler",
        "--mode",
        "-m",
        help="Emitter mode: sampler, embedding, or hybrid",
    ),
    iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of QD iterations",
    ),
    target_model: str = typer.Option(
        "gpt-5-mini",
        "--target",
        help="Target model to elicit behaviors from",
    ),
    judge_model: str = typer.Option(
        "gpt-5-mini",
        "--judge",
        help="Model to use for judging",
    ),
    sampler_model: Optional[str] = typer.Option(
        None,
        "--sampler-model",
        help="Model to use for prompt generation (sampler/hybrid modes). Defaults to target model.",
    ),
    rubric: Optional[Path] = typer.Option(
        None,
        "--rubric",
        "-r",
        help="Path to custom rubric template",
    ),
    score_coherence: bool = typer.Option(
        False,
        "--coherence",
        help="Also score prompt coherence",
    ),
    coherence_weight: float = typer.Option(
        0.0,
        "--coherence-weight",
        help="Weight for coherence in final score",
    ),
    requests_per_minute: int = typer.Option(
        100,
        "--rpm",
        help="Rate limit: requests per minute",
    ),
    tokens_per_minute: int = typer.Option(
        100_000,
        "--tpm",
        help="Rate limit: tokens per minute",
    ),
    max_concurrency: int = typer.Option(
        20,
        "--concurrency",
        "-c",
        help="Maximum concurrent API requests",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Temperature for target model generation",
    ),
    device: str = typer.Option(
        None,
        "--device",
        help="Device to use for embedding model",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint",
    ),
):
    """Run a behavior elicitation experiment."""
    # Map mode string to enum
    mode_map = {
        "sampler": EmitterMode.SAMPLER,
        "embedding": EmitterMode.EMBEDDING,
        "hybrid": EmitterMode.HYBRID,
    }
    if mode not in mode_map:
        console.print(
            f"[red]Invalid mode: {mode}. Use sampler, embedding, or hybrid.[/red]"
        )
        raise typer.Exit(1)

    # Build config
    config = BehaviorQDConfig(
        behavior_description=behavior,
        output_dir=output_dir,
    )
    config.scheduler.emitter_mode = mode_map[mode]
    config.scheduler.iterations = iterations
    config.target.model = target_model
    config.judge.model = judge_model
    config.judge.score_coherence = score_coherence
    config.judge.coherence_weight = coherence_weight
    config.target.temperature = temperature
    config.embedding.temperature = temperature
    config.embedding.device = device

    # Rate limit configuration
    config.rate_limit.requests_per_minute = requests_per_minute
    config.rate_limit.tokens_per_minute = tokens_per_minute
    config.rate_limit.max_concurrency = max_concurrency

    # Sampler model (only relevant for sampler/hybrid modes)
    if sampler_model:
        config.sampler.model = sampler_model

    if rubric:
        config.judge.rubric_path = rubric

    # Import here to avoid slow startup for help commands
    from flashlite import Flashlite
    from flashlite import RateLimitConfig as FlashliteRateLimitConfig

    from behavior_qd.scheduler import BehaviorQDScheduler

    # Create shared client for rate limiting across all components
    client = Flashlite(
        rate_limit=FlashliteRateLimitConfig(
            requests_per_minute=config.rate_limit.requests_per_minute,
            tokens_per_minute=config.rate_limit.tokens_per_minute,
        ),
        track_costs=True,
        cache=DiskCache(
            str(output_dir / "cache" / "completions.db"), default_ttl=86400
        ),
    )

    scheduler = BehaviorQDScheduler(config, client=client, console=console)

    if checkpoint:
        stats = scheduler.resume(checkpoint)
    else:
        stats = scheduler.run()

    # Save final visualizations
    from behavior_qd.visualization import save_all_visualizations

    save_all_visualizations(
        scheduler.archive,
        output_dir / "visualizations",
        stats.to_dict()["iterations"],
    )

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


@app.command("generate-rubric")
def generate_rubric(
    behavior: str = typer.Argument(
        ...,
        help="Description of the behavior (e.g., 'elicit sycophancy')",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for the rubric template",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="Model to use for generating the rubric",
    ),
    coherence: bool = typer.Option(
        False,
        "--coherence",
        help="Include coherence scoring in the rubric",
    ),
    freeform: bool = typer.Option(
        False,
        "--freeform",
        "-f",
        help="Generate freeform prose rubric instead of structured",
    ),
):
    """Generate a scoring rubric from a behavior description."""
    from behavior_qd.rubric import GeneratedRubric, RubricGenerator, RubricMode

    # Default output path
    if output is None:
        # Create a filename from the behavior
        safe_name = behavior.lower().replace(" ", "_")[:30]
        output = Path(f"rubrics/{safe_name}.jinja")

    mode = RubricMode.FREEFORM if freeform else RubricMode.STRUCTURED
    mode_label = "freeform" if freeform else "structured"

    console.print(f"[bold]Generating {mode_label} rubric for:[/bold] {behavior}")
    console.print(f"[dim]Using model: {model}[/dim]")

    generator = RubricGenerator(model=model)

    with console.status("Generating rubric..."):
        rubric = generator.generate_and_save_sync(
            behavior,
            output,
            include_coherence=coherence,
            mode=mode,
        )

    console.print(f"\n[green]Rubric saved to: {output}[/green]")

    # Display summary based on mode
    if isinstance(rubric, GeneratedRubric):
        # Structured mode - show detailed summary
        table = Table(title="Generated Rubric Summary")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Behavior Name", rubric.behavior_name)
        table.add_row("Definition", rubric.behavior_definition[:100] + "...")
        table.add_row("Scoring Levels", str(len(rubric.scoring_levels)))
        table.add_row("Examples", str(len(rubric.examples)))
        table.add_row("Edge Cases", str(len(rubric.edge_cases)))

        console.print(table)
    else:
        # Freeform mode - show preview
        console.print("\n[bold]Rubric Preview:[/bold]")
        preview = rubric[:500] + "..." if len(rubric) > 500 else rubric
        console.print(f"[dim]{preview}[/dim]")


@app.command("visualize")
def visualize(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for visualizations",
    ),
    stats_file: Optional[Path] = typer.Option(
        None,
        "--stats",
        help="Path to stats.json file",
    ),
):
    """Generate visualizations from a checkpoint."""
    from behavior_qd.archive import BehaviorArchive
    from behavior_qd.config import EmbeddingConfig
    from behavior_qd.embeddings import EmbeddingSpace
    from behavior_qd.visualization import save_all_visualizations

    if output_dir is None:
        output_dir = checkpoint.parent / "visualizations"

    console.print(f"[bold]Loading checkpoint:[/bold] {checkpoint}")

    # Load embedding space and archive
    with console.status("Loading embedding model..."):
        embedding_space = EmbeddingSpace(EmbeddingConfig())

    with console.status("Loading archive..."):
        archive = BehaviorArchive.load(checkpoint, embedding_space)

    # Load stats if available
    stats_iterations = None
    if stats_file and stats_file.exists():
        with open(stats_file) as f:
            data = json.load(f)
            stats_iterations = data.get("iterations")

    # Generate visualizations
    with console.status("Generating visualizations..."):
        save_all_visualizations(archive, output_dir, stats_iterations)

    console.print(f"\n[green]Visualizations saved to: {output_dir}[/green]")

    # Show archive stats
    stats = archive.stats
    table = Table(title="Archive Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Elites", str(stats["num_elites"]))
    table.add_row("Coverage", f"{stats['coverage']:.1%}")
    table.add_row("QD Score", f"{stats['qd_score']:.2f}")
    table.add_row("Best Objective", f"{stats['obj_max']:.3f}")
    table.add_row("Mean Objective", f"{stats['obj_mean']:.3f}")

    console.print(table)


@app.command("show-prompts")
def show_prompts(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
    ),
    n: int = typer.Option(
        10,
        "--num",
        "-n",
        help="Number of top prompts to show",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        "-e",
        help="Export prompts to file (json, csv, or txt)",
    ),
):
    """Show the best prompts from an archive."""
    from behavior_qd.archive import BehaviorArchive
    from behavior_qd.config import EmbeddingConfig
    from behavior_qd.embeddings import EmbeddingSpace

    console.print(f"[bold]Loading checkpoint:[/bold] {checkpoint}")

    with console.status("Loading..."):
        embedding_space = EmbeddingSpace(EmbeddingConfig())
        archive = BehaviorArchive.load(checkpoint, embedding_space)

    elites = archive.get_elites(n)

    console.print(f"\n[bold]Top {len(elites)} Prompts:[/bold]\n")

    for i, entry in enumerate(elites, 1):
        console.print(f"[bold cyan]{i}. Score: {entry.objective:.3f}[/bold cyan]")
        console.print(f"   [yellow]{entry.prompt}[/yellow]")
        if entry.response:
            response_preview = entry.response[:150].replace("\n", " ")
            console.print(f"   [dim]Response: {response_preview}...[/dim]")
        console.print()

    if export:
        from behavior_qd.visualization import export_best_prompts

        # Determine format from extension
        suffix = export.suffix.lower()
        format_map = {".json": "json", ".csv": "csv", ".txt": "txt"}
        fmt = format_map.get(suffix, "json")

        export_best_prompts(archive, export, n=n, format=fmt)
        console.print(f"[green]Exported to: {export}[/green]")


@app.command("evaluate")
def evaluate_prompt(
    prompt: str = typer.Argument(
        ...,
        help="Prompt to evaluate",
    ),
    behavior: str = typer.Option(
        ...,
        "--behavior",
        "-b",
        help="Behavior description to evaluate against",
    ),
    target_model: str = typer.Option(
        "gpt-5-mini",
        "--target",
        help="Target model",
    ),
    judge_model: str = typer.Option(
        "gpt-5-mini",
        "--judge",
        help="Judge model",
    ),
    rubric: Optional[Path] = typer.Option(
        None,
        "--rubric",
        "-r",
        help="Path to custom rubric",
    ),
):
    """Evaluate a single prompt against a behavior."""
    from behavior_qd.config import BehaviorQDConfig
    from behavior_qd.evaluation import Evaluator

    config = BehaviorQDConfig(behavior_description=behavior)
    config.target.model = target_model
    config.judge.model = judge_model

    if rubric:
        config.judge.rubric_path = rubric

    evaluator = Evaluator(config)

    console.print(f"[bold]Evaluating prompt:[/bold] {prompt}")
    console.print(f"[dim]Behavior: {behavior}[/dim]\n")

    with console.status("Evaluating..."):
        result = evaluator.evaluate_sync(prompt, behavior)

    console.print(
        f"[bold green]Behavior Score:[/bold green] {result.behavior_score:.3f}"
    )
    if result.coherence_score is not None:
        console.print(
            f"[bold blue]Coherence Score:[/bold blue] {result.coherence_score:.3f}"
        )
    console.print(f"[bold]Final Score:[/bold] {result.final_score:.3f}")
    console.print(f"\n[bold]Response:[/bold]\n{result.response}")
    console.print(f"\n[bold]Judge Reasoning:[/bold]\n{result.reasoning}")


if __name__ == "__main__":
    app()
