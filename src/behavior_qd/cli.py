"""Command-line interface for behavior-qd framework."""

import json
from pathlib import Path
from typing import Optional

import typer
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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from latest checkpoint in output directory",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from a specific checkpoint file (single-turn only)",
    ),
    seed_file: Optional[Path] = typer.Option(
        None,
        "--seed",
        help="CSV file with seed prompts (must have 'prompt' column)",
    ),
    turns: int = typer.Option(
        1,
        "--turns",
        "-t",
        help="Number of conversation turns (1 = single-turn, 2+ = multi-turn)",
    ),
    num_parents: Optional[int] = typer.Option(
        None,
        "--num-parents",
        help="Number of parent conversations to carry forward per turn "
        "(default: all elites)",
    ),
    continuation_iterations: int = typer.Option(
        50,
        "--continuation-iterations",
        help="QD iterations per continuation turn (Turn 2+)",
    ),
    response_max_tokens: int = typer.Option(
        1024,
        "--response-max-tokens",
        help="Max tokens for target responses in Turn 1+ (higher for full responses)",
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

    # Multi-turn configuration
    config.multi_turn.num_turns = turns
    config.multi_turn.response_max_tokens = response_max_tokens
    config.multi_turn.continuation_iterations = continuation_iterations
    if num_parents is not None:
        config.multi_turn.num_parent_conversations = num_parents

    # Sampler model (only relevant for sampler/hybrid modes)
    if sampler_model:
        config.sampler.model = sampler_model

    if rubric:
        config.judge.rubric_path = rubric

    if seed_file:
        config.scheduler.seed_file = seed_file

    # Import here to avoid slow startup for help commands
    from flashlite import Flashlite
    from flashlite import RateLimitConfig as FlashliteRateLimitConfig

    # Create shared client for rate limiting across all components
    client = Flashlite(
        rate_limit=FlashliteRateLimitConfig(
            requests_per_minute=config.rate_limit.requests_per_minute,
            tokens_per_minute=config.rate_limit.tokens_per_minute,
        ),
        track_costs=True,
    )

    if turns > 1:
        # Multi-turn mode
        from behavior_qd.multi_turn import MultiTurnScheduler

        scheduler = MultiTurnScheduler(config, client=client, console=console)

        if resume or checkpoint:
            _result = scheduler.resume()  # noqa: F841
        else:
            _result = scheduler.run()  # noqa: F841

        scheduler.save_all()

        # Save Turn 0 visualizations
        if scheduler._turn0_archive:
            from behavior_qd.visualization import save_all_visualizations

            save_all_visualizations(
                scheduler._turn0_archive,
                output_dir / "visualizations" / "turn0",
            )
    else:
        # Single-turn mode (original behavior)
        from behavior_qd.scheduler import BehaviorQDScheduler

        single_scheduler = BehaviorQDScheduler(config, client=client, console=console)

        if resume or checkpoint:
            stats = single_scheduler.resume(checkpoint)
        else:
            stats = single_scheduler.run()

        from behavior_qd.visualization import save_all_visualizations

        save_all_visualizations(
            single_scheduler.archive,
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
        help="Path to checkpoint file (Turn 0 .pkl or Turn 2+ continuation .pkl)",
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
    show_conversation: bool = typer.Option(
        False,
        "--conversation",
        help="Show full conversation thread for multi-turn entries",
    ),
    full_content: bool = typer.Option(
        False,
        "--full",
        help="Show full content without truncation",
    ),
):
    """Show the best prompts from an archive."""
    from behavior_qd.archive import BehaviorArchive, ContinuationArchive
    from behavior_qd.config import EmbeddingConfig
    from behavior_qd.embeddings import EmbeddingSpace

    console.print(f"[bold]Loading checkpoint:[/bold] {checkpoint}")

    with console.status("Loading..."):
        embedding_space = EmbeddingSpace(EmbeddingConfig())

        # Try loading as ContinuationArchive first, fall back to BehaviorArchive
        try:
            archive = ContinuationArchive.load(checkpoint, embedding_space)
            is_continuation = True
        except (KeyError, TypeError):
            archive = BehaviorArchive.load(checkpoint, embedding_space)
            is_continuation = False

    elites = archive.get_elites(n)

    label = "Conversations" if is_continuation else "Prompts"
    console.print(f"\n[bold]Top {len(elites)} {label}:[/bold]\n")

    for i, entry in enumerate(elites, 1):
        console.print(f"[bold cyan]{i}. Score: {entry.objective:.3f}[/bold cyan]")

        if show_conversation and entry.conversation:
            console.print(
                f"   [dim](Turn {entry.conversation.num_turns} conversation)[/dim]"
            )
            for turn in entry.conversation.turns:
                role = "User" if turn.role == "user" else "Assistant"
                content = turn.content
                if not full_content:
                    content = content[:200].replace("\n", " ")
                if turn.role == "user":
                    console.print(f"   [yellow][{role}]: {content}[/yellow]")
                else:
                    console.print(f"   [dim][{role}]: {content}[/dim]")
        else:
            prompt_text = entry.prompt
            if not full_content:
                prompt_text = prompt_text[:200].replace("\n", " ")
            console.print(f"   [yellow]{prompt_text}[/yellow]")
            if entry.response:
                response_text = entry.response
                if not full_content:
                    response_text = response_text[:150].replace("\n", " ") + "..."
                console.print(f"   [dim]Response: {response_text}[/dim]")
        console.print()

    if export:
        from behavior_qd.visualization import export_best_prompts

        suffix = export.suffix.lower()
        format_map = {".json": "json", ".csv": "csv", ".txt": "txt"}
        fmt = format_map.get(suffix, "json")

        export_best_prompts(archive, export, n=n, format=fmt)
        console.print(f"[green]Exported to: {export}[/green]")


@app.command("export-conversations")
def export_conversations(
    archive_path: Path = typer.Argument(
        ...,
        help="Path to archive file (.pkl) — continuation archive, turn1_parents.pkl, "
        "or behavior archive",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Defaults to conversations.json in same directory.",
    ),
    n: Optional[int] = typer.Option(
        None,
        "--num",
        "-n",
        help="Limit to top N conversations by score (default: all)",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json, jsonl, or txt",
    ),
    full_content: bool = typer.Option(
        False,
        "--full",
        help="Include full content without truncation (for txt format)",
    ),
):
    """Export conversations from an archive to a file.

    Supports:
    - Turn 2+ continuation archives (turnN_archive.pkl, turnN_checkpoint.pkl)
    - Turn 1 parent conversations (turn1_parents.pkl)
    - Turn 0 behavior archives (turn0_archive.pkl) — exports as prompt/response pairs
    """
    import json
    import pickle

    from behavior_qd.archive import BehaviorArchive, ContinuationArchive
    from behavior_qd.config import EmbeddingConfig
    from behavior_qd.embeddings import EmbeddingSpace

    console.print(f"[bold]Loading:[/bold] {archive_path}")

    # Determine output path
    if output is None:
        output = archive_path.parent / "conversations.json"

    entries = []
    archive_type = "unknown"

    with console.status("Loading archive..."):
        # Try turn1_parents.pkl (dict of PromptEntry)
        if "turn1_parents" in archive_path.name:
            with open(archive_path, "rb") as f:
                parent_dict = pickle.load(f)
            entries = list(parent_dict.values())
            archive_type = "turn1_parents"

        else:
            # Need embedding space for archive loading
            embedding_space = EmbeddingSpace(EmbeddingConfig())

            # Try ContinuationArchive first
            try:
                archive = ContinuationArchive.load(archive_path, embedding_space)
                entries = archive.get_elites(n)
                archive_type = "continuation"
            except (KeyError, TypeError):
                # Fall back to BehaviorArchive
                archive = BehaviorArchive.load(archive_path, embedding_space)
                entries = archive.get_elites(n)
                archive_type = "behavior"

    # Sort by objective (descending) and limit
    entries = sorted(entries, key=lambda e: e.objective, reverse=True)
    if n is not None:
        entries = entries[:n]

    console.print(
        f"[green]Loaded {len(entries)} entries[/green] (archive type: {archive_type})"
    )

    # Convert to export format
    conversations_data = []
    for i, entry in enumerate(entries):
        conv_data = {
            "rank": i + 1,
            "score": entry.objective,
        }

        if entry.conversation:
            conv_data["num_turns"] = entry.conversation.num_turns
            conv_data["turns"] = [
                {"role": t.role, "content": t.content} for t in entry.conversation.turns
            ]
            if entry.conversation.parent_id is not None:
                conv_data["parent_id"] = entry.conversation.parent_id
        else:
            # Single-turn: build conversation from prompt/response
            conv_data["num_turns"] = 1
            turns = [{"role": "user", "content": entry.prompt}]
            if entry.response:
                turns.append({"role": "assistant", "content": entry.response})
            conv_data["turns"] = turns

        if entry.reasoning:
            conv_data["judge_reasoning"] = entry.reasoning

        conversations_data.append(conv_data)

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output, "w") as f:
            json.dump(conversations_data, f, indent=2)

    elif format == "jsonl":
        with open(output, "w") as f:
            for conv in conversations_data:
                f.write(json.dumps(conv) + "\n")

    elif format == "txt":
        with open(output, "w") as f:
            for conv in conversations_data:
                f.write(f"{'=' * 60}\n")
                f.write(f"Rank: {conv['rank']} | Score: {conv['score']:.3f}\n")
                f.write(f"Turns: {conv['num_turns']}\n")
                f.write(f"{'=' * 60}\n\n")

                for turn in conv["turns"]:
                    role_label = "USER" if turn["role"] == "user" else "ASSISTANT"
                    content = turn["content"]
                    if not full_content and len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    f.write(f"[{role_label}]\n{content}\n\n")

                if "judge_reasoning" in conv:
                    reasoning = conv["judge_reasoning"]
                    if not full_content and len(reasoning) > 300:
                        reasoning = reasoning[:300] + "..."
                    f.write(f"[JUDGE REASONING]\n{reasoning}\n\n")

                f.write("\n")

    else:
        console.print(f"[red]Unknown format: {format}. Use json, jsonl, or txt.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]Exported {len(conversations_data)} conversations to:[/green]"
    )
    console.print(f"  {output}")


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
