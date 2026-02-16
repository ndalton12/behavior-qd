"""Scheduler for orchestrating the QD optimization loop."""

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flashlite import Flashlite
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from behavior_qd.archive import BehaviorArchive
from behavior_qd.config import BehaviorQDConfig, EmitterMode
from behavior_qd.embeddings import EmbeddingSpace
from behavior_qd.emitters import EmbeddingEmitter, SamplerEmitter
from behavior_qd.emitters.base import BaseEmitter, EmitterFeedback
from behavior_qd.emitters.embedding import HybridEmitter
from behavior_qd.evaluation import Evaluator


@dataclass
class IterationStats:
    """Statistics for a single iteration."""

    iteration: int
    num_evaluated: int
    num_added: int
    num_improved: int
    best_score: float
    mean_score: float
    archive_size: int
    archive_coverage: float
    qd_score: float
    duration_seconds: float
    total_cost: float = 0.0


@dataclass
class RunStats:
    """Statistics for an entire run."""

    iterations: list[IterationStats] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_evaluations: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "iterations": [
                {
                    "iteration": s.iteration,
                    "num_evaluated": s.num_evaluated,
                    "num_added": s.num_added,
                    "num_improved": s.num_improved,
                    "best_score": s.best_score,
                    "mean_score": s.mean_score,
                    "archive_size": s.archive_size,
                    "archive_coverage": s.archive_coverage,
                    "qd_score": s.qd_score,
                    "duration_seconds": s.duration_seconds,
                    "total_cost": s.total_cost,
                }
                for s in self.iterations
            ],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_evaluations": self.total_evaluations,
            "total_cost": self.total_cost,
        }


class BehaviorQDScheduler:
    """Orchestrates the QD optimization loop for behavior elicitation."""

    def __init__(
        self,
        config: BehaviorQDConfig,
        client: Flashlite,
        console: Console | None = None,
    ):
        """Initialize the scheduler.

        Args:
            config: Main configuration object.
            client: Shared Flashlite client for all API calls.
            console: Rich console for output.
        """
        self.config = config
        self.client = client
        self.console = console or Console()

        # Initialize components
        self._embedding_space: EmbeddingSpace | None = None
        self._archive: BehaviorArchive | None = None
        self._emitter: BaseEmitter | None = None
        self._evaluator: Evaluator | None = None

        # Run state
        self._stats = RunStats()
        self._iteration = 0
        self._resume_iteration = 0

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Lazily initialize embedding space."""
        if self._embedding_space is None:
            self.console.print("[dim]Loading embedding model...[/dim]")
            self._embedding_space = EmbeddingSpace(self.config.embedding)
            # Trigger lazy loading
            _ = self._embedding_space.pca
            self.console.print(
                f"[green]Loaded {self.config.embedding.model_name}[/green] "
                f"(dim={self._embedding_space.embed_dim}, vocab={self._embedding_space.vocab_size})"
            )
        return self._embedding_space

    @property
    def archive(self) -> BehaviorArchive:
        """Lazily initialize archive."""
        if self._archive is None:
            self._archive = BehaviorArchive(
                self.config.archive,
                self.embedding_space,
            )
        return self._archive

    @property
    def evaluator(self) -> Evaluator:
        """Lazily initialize evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(self.config, client=self.client)
        return self._evaluator

    @property
    def emitter(self) -> BaseEmitter:
        """Lazily initialize emitter based on config."""
        if self._emitter is None:
            mode = self.config.scheduler.emitter_mode

            if mode == EmitterMode.SAMPLER:
                self._emitter = SamplerEmitter(
                    archive=self.archive,
                    config=self.config.sampler,
                    behavior_description=self.config.behavior_description,
                    client=self.client,
                    template_dir=self.config.template_dir,
                )

            elif mode == EmitterMode.EMBEDDING:
                self._emitter = EmbeddingEmitter(
                    archive=self.archive,
                    embedding_space=self.embedding_space,
                    emitter_config=self.config.embedding_emitter,
                    embedding_config=self.config.embedding,
                )

            elif mode == EmitterMode.HYBRID:
                sampler = SamplerEmitter(
                    archive=self.archive,
                    config=self.config.sampler,
                    behavior_description=self.config.behavior_description,
                    client=self.client,
                    template_dir=self.config.template_dir,
                )
                embedding = EmbeddingEmitter(
                    archive=self.archive,
                    embedding_space=self.embedding_space,
                    emitter_config=self.config.embedding_emitter,
                    embedding_config=self.config.embedding,
                )
                self._emitter = HybridEmitter(
                    archive=self.archive,
                    sampler_emitter=sampler,
                    embedding_emitter=embedding,
                )

        return self._emitter

    def _run_iteration(self) -> IterationStats:
        """Run a single iteration of the QD loop.

        Returns:
            IterationStats for this iteration.
        """
        start_time = time.time()

        # Ask emitter for candidate prompts
        result = self.emitter.ask()
        prompts = result.prompts

        # Evaluate all prompts
        eval_results = self.evaluator.evaluate_batch_sync(
            prompts,
            behavior_description=self.config.behavior_description,
        )

        # Compute measures and add to archive
        measures_list = [self.embedding_space.compute_measures(p) for p in prompts]
        objectives = [r.final_score for r in eval_results]

        # Add to archive
        statuses = self.archive.add_batch(
            prompts=prompts,
            objectives=objectives,
            measures_list=measures_list,
            responses=[r.response for r in eval_results],
            reasonings=[r.reasoning for r in eval_results],
        )

        # Count additions and improvements
        # In pyribs 0.9+: NOT_ADDED=0, NEW=1, IMPROVE=2
        num_added = sum(1 for s in statuses if int(s) == 1)  # NEW
        num_improved = sum(1 for s in statuses if int(s) == 2)  # IMPROVE

        # Provide feedback to emitter
        feedback = EmitterFeedback(
            prompts=prompts,
            objectives=objectives,
            measures=measures_list,
            statuses=[int(s) for s in statuses],
        )
        self.emitter.tell(feedback)

        duration = time.time() - start_time

        # Compute stats
        archive_stats = self.archive.stats

        return IterationStats(
            iteration=self._iteration,
            num_evaluated=len(prompts),
            num_added=num_added,
            num_improved=num_improved,
            best_score=max(objectives) if objectives else 0.0,
            mean_score=sum(objectives) / len(objectives) if objectives else 0.0,
            archive_size=archive_stats["num_elites"],
            archive_coverage=archive_stats["coverage"],
            qd_score=archive_stats["qd_score"],
            duration_seconds=duration,
            total_cost=self.evaluator.total_cost,
        )

    def _log_iteration(self, stats: IterationStats) -> None:
        """Log iteration statistics.

        Args:
            stats: Statistics for the iteration.
        """
        self.console.print(
            f"[bold]Iter {stats.iteration:4d}[/bold] | "
            f"Eval: {stats.num_evaluated:3d} | "
            f"Added: {stats.num_added:3d} | "
            f"Improved: {stats.num_improved:3d} | "
            f"Best: {stats.best_score:.3f} | "
            f"Mean: {stats.mean_score:.3f} | "
            f"Archive: {stats.archive_size:4d} ({stats.archive_coverage:.1%}) | "
            f"QD: {stats.qd_score:.2f} | "
            f"Time: {stats.duration_seconds:.1f}s | "
            f"Cost: ${stats.total_cost:.4f}"
        )

    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current state."""
        checkpoint_path = self.config.get_output_path(
            f"checkpoint_iter{self._iteration}.pkl"
        )
        self.archive.save(checkpoint_path)

        # Also save stats
        stats_path = self.config.get_output_path("stats.json")
        with open(stats_path, "w") as f:
            json.dump(self._stats.to_dict(), f, indent=2)

        self.console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]")

    def _seed_archive(self) -> int:
        """Seed the archive with prompts from a CSV file.

        Returns:
            Number of prompts successfully added to the archive.
        """
        seed_file = self.config.scheduler.seed_file
        if seed_file is None:
            return 0

        self.console.print(f"[dim]Loading seed prompts from {seed_file}...[/dim]")

        # Read prompts from CSV
        prompts = []
        with open(seed_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "prompt" not in reader.fieldnames:
                self.console.print(
                    "[red]Error: CSV file must have a 'prompt' column[/red]"
                )
                return 0
            for row in reader:
                prompt = row["prompt"].strip()
                if prompt:
                    prompts.append(prompt)

        if not prompts:
            self.console.print("[yellow]No prompts found in seed file[/yellow]")
            return 0

        self.console.print(f"[dim]Evaluating {len(prompts)} seed prompts...[/dim]")

        # Evaluate all seed prompts
        eval_results = self.evaluator.evaluate_batch_sync(
            prompts,
            behavior_description=self.config.behavior_description,
        )

        # Compute measures and add to archive
        measures_list = [self.embedding_space.compute_measures(p) for p in prompts]
        objectives = [r.final_score for r in eval_results]

        # Add to archive
        statuses = self.archive.add_batch(
            prompts=prompts,
            objectives=objectives,
            measures_list=measures_list,
            responses=[r.response for r in eval_results],
            reasonings=[r.reasoning for r in eval_results],
        )

        # Count additions
        num_added = sum(1 for s in statuses if int(s) > 0)

        self.console.print(
            f"[green]Seeded archive with {num_added}/{len(prompts)} prompts[/green]"
        )

        return num_added

    def run(self) -> RunStats:
        """Run the full QD optimization loop.

        Supports both fresh runs and resumed runs.  When ``_resume_iteration``
        is > 0 (set by :meth:`resume`), the seed step is skipped, existing
        stats are preserved, and the loop starts from the resume point.

        Returns:
            RunStats with all iteration statistics.
        """
        is_resume = self._resume_iteration > 0
        total_iterations = self.config.scheduler.iterations
        start_iter = self._resume_iteration
        remaining = total_iterations - start_iter

        self.console.print("\n[bold blue]Starting Behavior QD Run[/bold blue]")
        self.console.print(f"Behavior: {self.config.behavior_description}")
        self.console.print(f"Mode: {self.config.scheduler.emitter_mode.value}")
        if is_resume:
            self.console.print(
                f"[bold yellow]Resuming from iteration {start_iter} "
                f"({remaining} remaining of {total_iterations})[/bold yellow]"
            )
        else:
            self.console.print(f"Iterations: {total_iterations}")
        self.console.print()

        if remaining <= 0:
            self.console.print(
                "[yellow]Already at or past target iterations — nothing to do[/yellow]"
            )
            if self._stats.end_time is None:
                self._stats.end_time = datetime.now()
            self._print_summary()
            return self._stats

        # Initialize components (triggers lazy loading)
        _ = self.embedding_space
        _ = self.archive
        _ = self.evaluator  # Initialize evaluator before seeding

        if not is_resume:
            # Only seed and reset stats for fresh runs
            self._seed_archive()
            self._stats = RunStats()

        # Initialize emitter after seeding (so it can use seeded elites)
        _ = self.emitter

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"{'Resuming' if is_resume else 'Running'} QD optimization...",
                total=remaining,
            )

            for i in range(remaining):
                self._iteration = start_iter + i + 1

                # Run iteration
                stats = self._run_iteration()
                self._stats.iterations.append(stats)
                self._stats.total_evaluations += stats.num_evaluated
                self._stats.total_cost = stats.total_cost

                # Log if needed
                if self._iteration % self.config.scheduler.log_interval == 0:
                    self._log_iteration(stats)

                # Checkpoint if needed
                if self._iteration % self.config.scheduler.checkpoint_interval == 0:
                    self._save_checkpoint()

                progress.update(task, advance=1)

        self._stats.end_time = datetime.now()

        # Final checkpoint
        self._save_checkpoint()

        # Print summary
        self._print_summary()

        # Reset resume state so a second call to run() starts fresh
        self._resume_iteration = 0

        return self._stats

    def _print_summary(self) -> None:
        """Print a summary of the run."""
        self.console.print("\n[bold green]Run Complete![/bold green]\n")

        # Summary table
        table = Table(title="Run Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        stats = self.archive.stats
        table.add_row("Total Iterations", str(len(self._stats.iterations)))
        table.add_row("Total Evaluations", str(self._stats.total_evaluations))
        table.add_row("Archive Size", str(stats["num_elites"]))
        table.add_row("Archive Coverage", f"{stats['coverage']:.1%}")
        table.add_row("QD Score", f"{stats['qd_score']:.2f}")
        table.add_row("Best Objective", f"{stats['obj_max']:.3f}")
        table.add_row("Mean Objective", f"{stats['obj_mean']:.3f}")
        table.add_row("Total Cost", f"${self._stats.total_cost:.4f}")

        if self._stats.end_time is not None:
            duration = (
                self._stats.end_time - self._stats.start_time
            ).total_seconds()
            table.add_row("Total Duration", f"{duration:.1f}s")
        else:
            table.add_row("Total Duration", "–")

        self.console.print(table)

        # Top prompts
        self.console.print("\n[bold]Top 5 Prompts:[/bold]")
        for i, entry in enumerate(self.archive.get_elites(5), 1):
            self.console.print(f"\n{i}. [cyan]Score: {entry.objective:.3f}[/cyan]")
            self.console.print(f"   [yellow]{entry.prompt}[/yellow]")
            if entry.reasoning:
                self.console.print(f"   [dim]{entry.reasoning[:100]}...[/dim]")

    def resume(self, checkpoint_path: Path | str | None = None) -> RunStats:
        """Resume from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.  When *None*, the latest
                ``checkpoint_iter*.pkl`` in the output directory is used.

        Returns:
            RunStats continuing from checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                self.console.print(
                    "[yellow]No checkpoint found, starting fresh[/yellow]"
                )
                return self.run()

        checkpoint_path = Path(checkpoint_path)
        self.console.print(
            f"[dim]Loading checkpoint from {checkpoint_path}...[/dim]"
        )
        self._archive = BehaviorArchive.load(checkpoint_path, self.embedding_space)

        # Try to load stats from the checkpoint's directory, then output dir
        stats_loaded = False
        for search_dir in [checkpoint_path.parent, self.config.output_dir]:
            stats_path = search_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    data = json.load(f)
                    self._stats = RunStats()
                    self._stats.iterations = [
                        IterationStats(**s) for s in data["iterations"]
                    ]
                    self._stats.total_evaluations = sum(
                        s.num_evaluated for s in self._stats.iterations
                    )
                    if data.get("total_cost"):
                        self._stats.total_cost = data["total_cost"]
                    if data.get("start_time"):
                        self._stats.start_time = datetime.fromisoformat(
                            data["start_time"]
                        )
                    if data.get("end_time"):
                        self._stats.end_time = datetime.fromisoformat(
                            data["end_time"]
                        )
                    self._resume_iteration = len(self._stats.iterations)
                    stats_loaded = True
                    break

        if not stats_loaded:
            # Fall back: infer iteration from checkpoint filename
            import re

            match = re.search(r"checkpoint_iter(\d+)", checkpoint_path.name)
            if match:
                self._resume_iteration = int(match.group(1))
            else:
                self.console.print(
                    "[yellow]Could not determine iteration from checkpoint — "
                    "starting from iteration 0[/yellow]"
                )
                self._resume_iteration = 0

        self.console.print(
            f"[green]Loaded checkpoint: "
            f"{self.archive.stats['num_elites']} elites, "
            f"resuming from iteration {self._resume_iteration}[/green]"
        )

        return self.run()

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the most recent ``checkpoint_iter*.pkl`` in the output dir."""
        output_dir = self.config.output_dir
        if not output_dir.exists():
            return None

        checkpoints = sorted(
            output_dir.glob("checkpoint_iter*.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None
