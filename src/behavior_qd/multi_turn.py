"""Multi-turn scheduler for orchestrating conversation QD across turns.

Manages the pipeline: Turn 0 (zero-shot QD) → Turn 1 (response gathering)
→ Turn 2..N (continuation QD with increasing conversation depth).
"""

from __future__ import annotations

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

from behavior_qd.archive import BehaviorArchive, ContinuationArchive, PromptEntry
from behavior_qd.config import BehaviorQDConfig
from behavior_qd.conversation import Conversation
from behavior_qd.embeddings import EmbeddingSpace
from behavior_qd.emitters.base import EmitterFeedback
from behavior_qd.emitters.continuation import ContinuationSamplerEmitter
from behavior_qd.evaluation import Evaluator
from behavior_qd.scheduler import BehaviorQDScheduler, IterationStats, RunStats


@dataclass
class TurnResult:
    """Result from a single turn of multi-turn QD."""

    turn_number: int
    num_parents: int  # 0 for Turn 0
    run_stats: RunStats | None = None  # For QD turns (0, 2+)
    num_conversations: int = 0  # For Turn 1 (response gathering)
    archive_stats: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class MultiTurnResult:
    """Result from the full multi-turn pipeline."""

    turn_results: list[TurnResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_cost: float = 0.0


class MultiTurnScheduler:
    """Orchestrates multi-turn conversation QD.

    Pipeline:
    1. Turn 0: Standard zero-shot QD (via BehaviorQDScheduler)
    2. Turn 1: Gather full responses for Turn 0 elites (response collection)
    3. Turn 2..N: Continuation QD — generate follow-ups, evaluate full
       conversations, archive by parent + continuation diversity
    """

    def __init__(
        self,
        config: BehaviorQDConfig,
        client: Flashlite,
        console: Console | None = None,
    ):
        """Initialize the multi-turn scheduler.

        Args:
            config: Main configuration (includes multi_turn settings).
            client: Shared Flashlite client for all API calls.
            console: Rich console for output.
        """
        self.config = config
        self.client = client
        self.console = console or Console()

        # Shared components
        self._embedding_space: EmbeddingSpace | None = None
        self._evaluator: Evaluator | None = None

        # Per-turn archives
        self._turn0_archive: BehaviorArchive | None = None
        self._turn_archives: dict[int, ContinuationArchive] = {}

        # Turn 1 data: parent conversations indexed by position
        self._parent_conversations: dict[int, PromptEntry] = {}

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Lazily initialize embedding space."""
        if self._embedding_space is None:
            self.console.print("[dim]Loading embedding model...[/dim]")
            self._embedding_space = EmbeddingSpace(self.config.embedding)
            _ = self._embedding_space.pca
            self.console.print(
                f"[green]Loaded {self.config.embedding.model_name}[/green] "
                f"(dim={self._embedding_space.embed_dim})"
            )
        return self._embedding_space

    @property
    def evaluator(self) -> Evaluator:
        """Lazily initialize evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(self.config, client=self.client)
        return self._evaluator

    # ------------------------------------------------------------------
    # Turn 0: Standard zero-shot QD
    # ------------------------------------------------------------------

    def _run_turn0(self) -> TurnResult:
        """Run Turn 0: standard zero-shot QD via BehaviorQDScheduler.

        Returns:
            TurnResult for Turn 0.
        """
        self.console.print("\n[bold blue]Turn 0: Zero-shot QD[/bold blue]")

        start = time.time()

        scheduler = BehaviorQDScheduler(
            config=self.config,
            client=self.client,
            console=self.console,
        )
        # Share the embedding space to avoid reloading
        scheduler._embedding_space = self.embedding_space

        run_stats = scheduler.run()
        self._turn0_archive = scheduler.archive

        duration = time.time() - start

        return TurnResult(
            turn_number=0,
            num_parents=0,
            run_stats=run_stats,
            archive_stats=self._turn0_archive.stats,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Turn 1: Response gathering
    # ------------------------------------------------------------------

    def _run_turn1(self) -> TurnResult:
        """Run Turn 1: gather full responses for Turn 0 elites.

        Sends each elite prompt to the target model with higher max_tokens,
        recording (prompt, full_response) pairs.

        Returns:
            TurnResult for Turn 1.
        """
        self.console.print("\n[bold blue]Turn 1: Gathering responses[/bold blue]")

        start = time.time()
        mt_config = self.config.multi_turn

        # Get elites from Turn 0
        num_parents = mt_config.num_parent_conversations
        elites = self._turn0_archive.get_elites(n=num_parents)

        if not elites:
            self.console.print("[red]No elites in Turn 0 archive![/red]")
            return TurnResult(turn_number=1, num_parents=0, duration_seconds=0.0)

        self.console.print(
            f"[dim]Gathering responses for {len(elites)} elite prompts "
            f"(max_tokens={mt_config.response_max_tokens})...[/dim]"
        )

        # Evaluate all prompts with higher max_tokens to get full responses
        prompts = [e.prompt for e in elites]
        eval_results = self.evaluator.evaluate_batch_sync(
            prompts=prompts,
            behavior_description=self.config.behavior_description,
            max_tokens=mt_config.response_max_tokens,
        )

        # Build parent conversation entries
        self._parent_conversations = {}
        for idx, (elite, eval_result) in enumerate(zip(elites, eval_results)):
            conversation = Conversation.from_prompt_response(
                prompt=elite.prompt,
                response=eval_result.response,
            )
            parent_entry = PromptEntry(
                prompt=elite.prompt,
                objective=eval_result.final_score,
                measures=elite.measures,
                response=eval_result.response,
                reasoning=eval_result.reasoning,
                conversation=conversation,
            )
            self._parent_conversations[idx] = parent_entry

        duration = time.time() - start

        self.console.print(
            f"[green]Gathered {len(self._parent_conversations)} "
            f"conversation pairs[/green] in {duration:.1f}s"
        )

        # Log some stats about response lengths
        resp_lengths = [
            len(e.response.split()) if e.response else 0
            for e in self._parent_conversations.values()
        ]
        if resp_lengths:
            self.console.print(
                f"[dim]Response lengths: "
                f"min={min(resp_lengths)}, "
                f"mean={sum(resp_lengths) / len(resp_lengths):.0f}, "
                f"max={max(resp_lengths)} words[/dim]"
            )

        return TurnResult(
            turn_number=1,
            num_parents=len(self._parent_conversations),
            num_conversations=len(self._parent_conversations),
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Turn 2+: Continuation QD
    # ------------------------------------------------------------------

    def _run_continuation_turn(
        self,
        turn_number: int,
        *,
        resume_archive: ContinuationArchive | None = None,
        resume_iteration: int = 0,
    ) -> TurnResult:
        """Run a continuation QD turn.

        Generates follow-up messages for parent conversations, evaluates
        the full conversation thread, and archives by parent + diversity.

        Args:
            turn_number: The turn number (2+).
            resume_archive: If provided, resume into this partially-filled
                archive instead of creating a new one.
            resume_iteration: Iteration to resume from (0 = fresh start).

        Returns:
            TurnResult for this turn.
        """
        is_resume = resume_iteration > 0
        self.console.print(
            f"\n[bold blue]Turn {turn_number}: Continuation QD"
            f"{' (resuming)' if is_resume else ''}[/bold blue]"
        )

        start = time.time()
        mt_config = self.config.multi_turn

        # Determine parent entries for this turn
        if turn_number == 2:
            parent_entries = self._parent_conversations
        else:
            # For Turn 3+, use elites from the previous continuation archive
            prev_archive = self._turn_archives.get(turn_number - 1)
            if prev_archive is None:
                self.console.print(
                    f"[red]No archive for Turn {turn_number - 1}![/red]"
                )
                return TurnResult(
                    turn_number=turn_number, num_parents=0, duration_seconds=0.0
                )

            prev_elites = prev_archive.get_elites(n=mt_config.num_parent_conversations)
            parent_entries = {idx: elite for idx, elite in enumerate(prev_elites)}

        num_parents = len(parent_entries)
        if num_parents == 0:
            self.console.print("[red]No parent conversations available![/red]")
            return TurnResult(
                turn_number=turn_number, num_parents=0, duration_seconds=0.0
            )

        total_iterations = mt_config.continuation_iterations
        remaining = total_iterations - resume_iteration

        if is_resume:
            self.console.print(
                f"[bold yellow]Resuming from iteration {resume_iteration} "
                f"({remaining} remaining of {total_iterations})[/bold yellow]"
            )
        else:
            self.console.print(
                f"[dim]Running continuation QD with {num_parents} parents, "
                f"{total_iterations} iterations[/dim]"
            )

        # Use resumed archive or create a fresh one
        if resume_archive is not None:
            cont_archive = resume_archive
        else:
            cont_archive = ContinuationArchive(
                num_parents=num_parents,
                embedding_space=self.embedding_space,
                archive_config=self.config.archive,
                multi_turn_config=mt_config,
            )
            cont_archive.set_parent_entries(parent_entries)
        self._turn_archives[turn_number] = cont_archive

        # Create continuation emitter
        parent_list = [parent_entries[i] for i in range(num_parents)]
        emitter = ContinuationSamplerEmitter(
            archive=cont_archive,
            config=self.config.sampler,
            behavior_description=self.config.behavior_description,
            parent_entries=parent_list,
            client=self.client,
            template_dir=self.config.template_dir,
        )

        # Run QD loop
        iteration_stats = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Turn {turn_number} QD{' (resumed)' if is_resume else ''}...",
                total=remaining,
            )

            for i in range(remaining):
                current_iter = resume_iteration + i + 1
                iter_stats = self._run_continuation_iteration(
                    emitter=emitter,
                    archive=cont_archive,
                    parent_entries=parent_entries,
                    iteration=current_iter,
                )
                iteration_stats.append(iter_stats)

                if current_iter % self.config.scheduler.log_interval == 0:
                    self._log_iteration(iter_stats, turn_number)

                if current_iter % self.config.scheduler.checkpoint_interval == 0:
                    self._save_turn_checkpoint(
                        turn_number, cont_archive, iteration=current_iter
                    )

                progress.update(task, advance=1)

        # Re-gather full responses for elites (QD used short max_tokens)
        self._regather_full_responses(cont_archive)

        # Final checkpoint (now includes full responses)
        self._save_turn_checkpoint(
            turn_number, cont_archive, iteration=total_iterations
        )

        duration = time.time() - start

        # Build run stats
        run_stats = RunStats(iterations=iteration_stats)
        run_stats.end_time = datetime.now()
        run_stats.total_evaluations = sum(s.num_evaluated for s in iteration_stats)
        run_stats.total_cost = self.evaluator.total_cost

        result = TurnResult(
            turn_number=turn_number,
            num_parents=num_parents,
            run_stats=run_stats,
            archive_stats=cont_archive.stats,
            duration_seconds=duration,
        )

        self._print_turn_summary(result, cont_archive)

        return result

    def _run_continuation_iteration(
        self,
        emitter: ContinuationSamplerEmitter,
        archive: ContinuationArchive,
        parent_entries: dict[int, PromptEntry],
        iteration: int,
    ) -> IterationStats:
        """Run a single iteration of continuation QD.

        Args:
            emitter: The continuation sampler emitter.
            archive: The continuation archive.
            parent_entries: Parent conversation entries.
            iteration: Current iteration number.

        Returns:
            IterationStats for this iteration.
        """
        start = time.time()

        # Ask emitter for candidate follow-ups
        result = emitter.ask()
        followups = result.prompts

        # Get parent info from result metadata
        parent_idx = getattr(result, "_parent_idx", 0)
        parent_entry = getattr(result, "_parent_entry", None)
        base_conversation = getattr(result, "_conversation", None)

        if parent_entry is None or base_conversation is None:
            parent_entry = parent_entries[parent_idx]
            base_conversation = self._get_conversation(parent_entry)

        # Build full conversations: base + each follow-up
        conversations = []
        for followup in followups:
            conv = base_conversation.with_continuation(followup)
            conversations.append(conv)

        # Evaluate full conversations (short max_tokens during QD for efficiency;
        # full responses are re-gathered for elites after the QD loop finishes)
        eval_results = self.evaluator.evaluate_batch_sync(
            prompts=followups,
            behavior_description=self.config.behavior_description,
            conversations=conversations,
        )

        # Compute measures on the continuation text (not the full conversation)
        measures_list = [
            self.embedding_space.compute_measures(f) for f in followups
        ]
        objectives = [r.final_score for r in eval_results]

        # Build complete conversations with responses for archiving
        full_conversations = []
        for conv, eval_result in zip(conversations, eval_results):
            full_conv = Conversation(
                turns=list(conv.turns),
                parent_id=parent_idx,
            )
            from behavior_qd.conversation import ConversationTurn

            full_conv.turns.append(
                ConversationTurn(role="assistant", content=eval_result.response)
            )
            full_conversations.append(full_conv)

        # Add to continuation archive
        statuses = archive.add_batch(
            prompts=followups,
            objectives=objectives,
            parent_indices=[parent_idx] * len(followups),
            measures_list=measures_list,
            responses=[r.response for r in eval_results],
            reasonings=[r.reasoning for r in eval_results],
            conversations=full_conversations,
        )

        num_added = sum(1 for s in statuses if s == 1)
        num_improved = sum(1 for s in statuses if s == 2)

        # Feedback to emitter
        feedback = EmitterFeedback(
            prompts=followups,
            objectives=objectives,
            measures=measures_list,
            statuses=statuses,
        )
        emitter.tell(feedback)

        duration = time.time() - start
        archive_stats = archive.stats

        return IterationStats(
            iteration=iteration,
            num_evaluated=len(followups),
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

    def _regather_full_responses(self, archive: ContinuationArchive) -> None:
        """Re-gather full-length responses for all elites in a continuation archive.

        During QD, we use the default (short) ``target.max_tokens`` for
        efficiency.  Once the QD loop finishes we re-send each elite's
        full conversation to the target model with the higher
        ``multi_turn.response_max_tokens`` and update the stored entries
        so downstream turns and exports see the complete responses.
        """
        elites = archive.get_elites()
        if not elites:
            return

        self.console.print(
            f"[dim]Re-gathering full responses for {len(elites)} elites "
            f"(max_tokens={self.config.multi_turn.response_max_tokens})...[/dim]"
        )

        # Build conversations up to the last user message (strip the
        # truncated assistant reply that was stored during QD)
        conversations_to_send: list[Conversation] = []
        for entry in elites:
            conv = self._get_conversation(entry)
            # If the last turn is an assistant message, remove it — we'll
            # replace it with the full-length response.
            if conv.turns and conv.turns[-1].role == "assistant":
                conv = Conversation(
                    turns=list(conv.turns[:-1]),
                    parent_id=conv.parent_id,
                )
            conversations_to_send.append(conv)

        # Re-evaluate with full max_tokens (only the target call matters;
        # we keep the existing judge scores from the QD run)
        prompts = [e.prompt for e in elites]
        eval_results = self.evaluator.evaluate_batch_sync(
            prompts=prompts,
            behavior_description=self.config.behavior_description,
            conversations=conversations_to_send,
            max_tokens=self.config.multi_turn.response_max_tokens,
        )

        # Update each elite's stored response and conversation in-place
        from behavior_qd.conversation import ConversationTurn

        for entry, conv_sent, eval_result in zip(
            elites, conversations_to_send, eval_results
        ):
            full_response = eval_result.response
            entry.response = full_response

            # Rebuild conversation: take the turns we sent (truncated
            # assistant already stripped) and append the full response
            if entry.conversation is not None:
                new_turns = list(conv_sent.turns) + [
                    ConversationTurn(role="assistant", content=full_response)
                ]
                entry.conversation = Conversation(
                    turns=new_turns,
                    parent_id=entry.conversation.parent_id,
                )

        self.console.print(
            f"[green]Updated {len(elites)} elites with full responses[/green]"
        )

    def _get_conversation(self, entry: PromptEntry) -> Conversation:
        """Get or build the conversation from a PromptEntry.

        Args:
            entry: The prompt entry.

        Returns:
            Conversation object.
        """
        if entry.conversation is not None:
            return entry.conversation
        return Conversation.from_prompt_response(entry.prompt, entry.response)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> MultiTurnResult:
        """Run the full multi-turn pipeline.

        Returns:
            MultiTurnResult with all turn results.
        """
        num_turns = self.config.multi_turn.num_turns
        total_start = time.time()

        self.console.print(
            f"\n[bold magenta]Multi-Turn Behavior QD[/bold magenta] "
            f"({num_turns} turn{'s' if num_turns > 1 else ''})"
        )
        self.console.print(f"Behavior: {self.config.behavior_description}")
        self.console.print()

        result = MultiTurnResult()

        # Turn 0: Zero-shot QD
        turn0_result = self._run_turn0()
        result.turn_results.append(turn0_result)
        self._save_completed_turn(0)

        if num_turns == 1:
            # Single-turn mode — we're done
            result.total_duration_seconds = time.time() - total_start
            result.total_cost = self.evaluator.total_cost
            self._print_final_summary(result)
            return result

        # Turn 1: Response gathering
        turn1_result = self._run_turn1()
        result.turn_results.append(turn1_result)
        self._save_completed_turn(1)

        # Turn 2..N: Continuation QD
        for turn in range(2, num_turns + 1):
            turn_result = self._run_continuation_turn(turn)
            result.turn_results.append(turn_result)
            self._save_completed_turn(turn)

        result.total_duration_seconds = time.time() - total_start
        result.total_cost = self.evaluator.total_cost
        self._print_final_summary(result)

        return result

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(self) -> MultiTurnResult:
        """Resume the multi-turn pipeline from the last completed turn.

        Auto-detects progress by checking which turn archives exist in the
        output directory:

        * ``turn0_archive.pkl`` → Turn 0 complete
        * ``turn1_parents.pkl`` → Turn 1 complete
        * ``turn{N}_archive.pkl`` → Turn N complete
        * ``turn{N}_checkpoint.pkl`` → Turn N was interrupted mid-QD

        If Turn 0 was interrupted (checkpoint exists but no final archive),
        it is resumed via :meth:`BehaviorQDScheduler.resume`.

        Returns:
            MultiTurnResult with results from all turns.
        """
        import pickle

        output_dir = self.config.output_dir
        num_turns = self.config.multi_turn.num_turns

        self.console.print(
            f"\n[bold magenta]Resuming Multi-Turn Behavior QD[/bold magenta] "
            f"({num_turns} turn{'s' if num_turns > 1 else ''})"
        )
        self.console.print(f"Behavior: {self.config.behavior_description}")
        self.console.print(f"Output dir: {output_dir}")
        self.console.print()

        total_start = time.time()
        result = MultiTurnResult()
        resume_from_turn = 0  # first turn that still needs work

        # ---- Detect completed turns ----

        # Turn 0
        turn0_archive_path = output_dir / "turn0_archive.pkl"
        turn0_checkpoint = self._find_latest_turn0_checkpoint()

        if turn0_archive_path.exists():
            self.console.print("[dim]Loading completed Turn 0 archive...[/dim]")
            self._turn0_archive = BehaviorArchive.load(
                turn0_archive_path, self.embedding_space
            )
            stats = self._turn0_archive.stats
            result.turn_results.append(
                TurnResult(
                    turn_number=0,
                    num_parents=0,
                    archive_stats=stats,
                )
            )
            resume_from_turn = 1
            self.console.print(
                f"[green]  Turn 0: loaded ({stats['num_elites']} elites)[/green]"
            )
        elif turn0_checkpoint:
            self.console.print(
                "[yellow]Turn 0 was interrupted — resuming from checkpoint...[/yellow]"
            )
            turn0_result = self._resume_turn0(turn0_checkpoint)
            result.turn_results.append(turn0_result)
            self._save_completed_turn(0)
            resume_from_turn = 1

        if resume_from_turn == 0:
            self.console.print(
                "[yellow]No previous state found — starting fresh[/yellow]"
            )
            return self.run()

        # Turn 1
        if num_turns > 1:
            turn1_path = output_dir / "turn1_parents.pkl"
            if turn1_path.exists():
                self.console.print("[dim]Loading completed Turn 1 parents...[/dim]")
                with open(turn1_path, "rb") as f:
                    self._parent_conversations = pickle.load(f)
                result.turn_results.append(
                    TurnResult(
                        turn_number=1,
                        num_parents=len(self._parent_conversations),
                        num_conversations=len(self._parent_conversations),
                    )
                )
                resume_from_turn = 2
                self.console.print(
                    f"[green]  Turn 1: loaded "
                    f"({len(self._parent_conversations)} parents)[/green]"
                )
            elif resume_from_turn >= 1:
                # Turn 1 hasn't completed yet — run it
                self.console.print("[dim]Running Turn 1...[/dim]")
                turn1_result = self._run_turn1()
                result.turn_results.append(turn1_result)
                self._save_completed_turn(1)
                resume_from_turn = 2

        # Turns 2+
        for turn in range(2, num_turns + 1):
            archive_path = output_dir / f"turn{turn}_archive.pkl"
            checkpoint_path = output_dir / f"turn{turn}_checkpoint.pkl"

            if archive_path.exists():
                self.console.print(
                    f"[dim]Loading completed Turn {turn} archive...[/dim]"
                )
                loaded = ContinuationArchive.load(archive_path, self.embedding_space)
                self._turn_archives[turn] = loaded
                stats = loaded.stats
                result.turn_results.append(
                    TurnResult(
                        turn_number=turn,
                        num_parents=loaded.num_parents,
                        archive_stats=stats,
                    )
                )
                self.console.print(
                    f"[green]  Turn {turn}: loaded "
                    f"({stats['num_elites']} elites)[/green]"
                )
            elif checkpoint_path.exists():
                # Turn was interrupted mid-QD — resume it
                self.console.print(
                    f"[yellow]Turn {turn} was interrupted — "
                    f"resuming from checkpoint...[/yellow]"
                )
                loaded_archive = ContinuationArchive.load(
                    checkpoint_path, self.embedding_space
                )
                saved_iter = self._load_turn_iteration(turn)
                self.console.print(
                    f"[dim]  Loaded checkpoint with "
                    f"{loaded_archive.stats['num_elites']} elites "
                    f"at iteration {saved_iter}[/dim]"
                )
                turn_result = self._run_continuation_turn(
                    turn,
                    resume_archive=loaded_archive,
                    resume_iteration=saved_iter,
                )
                result.turn_results.append(turn_result)
                self._save_completed_turn(turn)
            else:
                # Turn hasn't started — run fresh
                turn_result = self._run_continuation_turn(turn)
                result.turn_results.append(turn_result)
                self._save_completed_turn(turn)

        result.total_duration_seconds = time.time() - total_start
        result.total_cost = self.evaluator.total_cost
        self._print_final_summary(result)

        return result

    def _resume_turn0(self, checkpoint_path: Path) -> TurnResult:
        """Resume Turn 0 from an interrupted checkpoint.

        Uses :class:`BehaviorQDScheduler` to continue the zero-shot QD run.
        """
        start = time.time()

        scheduler = BehaviorQDScheduler(
            config=self.config,
            client=self.client,
            console=self.console,
        )
        # Share the embedding space so it doesn't reload
        scheduler._embedding_space = self.embedding_space

        run_stats = scheduler.resume(checkpoint_path)
        self._turn0_archive = scheduler.archive

        duration = time.time() - start
        return TurnResult(
            turn_number=0,
            num_parents=0,
            run_stats=run_stats,
            archive_stats=self._turn0_archive.stats,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Logging & checkpointing
    # ------------------------------------------------------------------

    def _log_iteration(self, stats: IterationStats, turn_number: int) -> None:
        """Log iteration statistics for a continuation turn."""
        self.console.print(
            f"[bold]T{turn_number} Iter {stats.iteration:4d}[/bold] | "
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

    def _save_completed_turn(self, turn_number: int) -> None:
        """Save a completed turn's archive for resume detection.

        Files saved per turn enable :meth:`resume` to skip already-finished
        turns:

        * Turn 0 → ``turn0_archive.pkl``
        * Turn 1 → ``turn1_parents.pkl``
        * Turn N (≥ 2) → ``turn{N}_archive.pkl``
        """
        import pickle

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if turn_number == 0 and self._turn0_archive:
            self._turn0_archive.save(output_dir / "turn0_archive.pkl")
        elif turn_number == 1 and self._parent_conversations:
            with open(output_dir / "turn1_parents.pkl", "wb") as f:
                pickle.dump(self._parent_conversations, f)
        elif turn_number >= 2 and turn_number in self._turn_archives:
            self._turn_archives[turn_number].save(
                output_dir / f"turn{turn_number}_archive.pkl"
            )

    def _save_turn_checkpoint(
        self,
        turn_number: int,
        archive: ContinuationArchive,
        iteration: int | None = None,
    ) -> None:
        """Save a mid-turn checkpoint with optional iteration state."""
        checkpoint_path = self.config.get_output_path(
            f"turn{turn_number}_checkpoint.pkl"
        )
        archive.save(checkpoint_path)

        # Persist the iteration counter so resume knows where to pick up
        if iteration is not None:
            state_path = self.config.get_output_path(
                f"turn{turn_number}_state.json"
            )
            with open(state_path, "w") as f:
                json.dump({"iteration": iteration, "turn_number": turn_number}, f)

        self.console.print(
            f"[dim]Saved Turn {turn_number} checkpoint"
            f"{f' (iter {iteration})' if iteration is not None else ''}[/dim]"
        )

    def _load_turn_iteration(self, turn_number: int) -> int:
        """Load the saved iteration count for a turn checkpoint.

        Returns 0 if no state file is found.
        """
        state_path = self.config.get_output_path(
            f"turn{turn_number}_state.json"
        )
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)
                return data.get("iteration", 0)
        return 0

    def _find_latest_turn0_checkpoint(self) -> Path | None:
        """Find the most recent Turn 0 checkpoint in the output dir."""
        output_dir = self.config.output_dir
        if not output_dir.exists():
            return None

        checkpoints = sorted(
            output_dir.glob("checkpoint_iter*.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None

    def _print_turn_summary(
        self, result: TurnResult, archive: ContinuationArchive
    ) -> None:
        """Print a summary for a completed turn."""
        stats = archive.stats

        self.console.print(
            f"\n[bold green]Turn {result.turn_number} Complete[/bold green]"
        )
        self.console.print(
            f"  Archive: {stats['num_elites']} elites, "
            f"{stats['coverage']:.1%} coverage, "
            f"QD={stats['qd_score']:.2f}"
        )

        # Show top conversations
        top_elites = archive.get_elites(3)
        for i, entry in enumerate(top_elites, 1):
            self.console.print(
                f"\n  {i}. [cyan]Score: {entry.objective:.3f}[/cyan]"
            )
            if entry.conversation:
                self.console.print(
                    f"     [dim]{entry.conversation.format_for_display(100)}[/dim]"
                )
            else:
                self.console.print(f"     [yellow]{entry.prompt[:100]}...[/yellow]")

    def _print_final_summary(self, result: MultiTurnResult) -> None:
        """Print the final multi-turn summary."""
        self.console.print(
            "\n[bold magenta]Multi-Turn Run Complete![/bold magenta]\n"
        )

        table = Table(title="Multi-Turn Summary")
        table.add_column("Turn", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Parents", style="yellow")
        table.add_column("Elites", style="green")
        table.add_column("QD Score", style="magenta")
        table.add_column("Duration", style="dim")

        for tr in result.turn_results:
            if tr.turn_number == 0:
                turn_type = "Zero-shot QD"
                elites = str(tr.archive_stats.get("num_elites", "–"))
                qd = f"{tr.archive_stats.get('qd_score', 0):.2f}"
            elif tr.turn_number == 1:
                turn_type = "Response gathering"
                elites = str(tr.num_conversations)
                qd = "–"
            else:
                turn_type = "Continuation QD"
                elites = str(tr.archive_stats.get("num_elites", "–"))
                qd = f"{tr.archive_stats.get('qd_score', 0):.2f}"

            table.add_row(
                str(tr.turn_number),
                turn_type,
                str(tr.num_parents),
                elites,
                qd,
                f"{tr.duration_seconds:.1f}s",
            )

        self.console.print(table)
        self.console.print(
            f"\nTotal Duration: {result.total_duration_seconds:.1f}s"
        )
        self.console.print(f"Total Cost: ${result.total_cost:.4f}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_all(self, output_dir: Path | None = None) -> None:
        """Save all turn archives and metadata.

        Args:
            output_dir: Output directory. Defaults to config output_dir.
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Turn 0 archive
        if self._turn0_archive:
            self._turn0_archive.save(output_dir / "turn0_archive.pkl")

        # Save parent conversations (Turn 1)
        if self._parent_conversations:
            import pickle

            with open(output_dir / "turn1_parents.pkl", "wb") as f:
                pickle.dump(self._parent_conversations, f)

        # Save continuation archives (Turn 2+)
        for turn_num, archive in self._turn_archives.items():
            archive.save(output_dir / f"turn{turn_num}_archive.pkl")

        self.console.print(
            f"[green]Saved all turn archives to {output_dir}[/green]"
        )
