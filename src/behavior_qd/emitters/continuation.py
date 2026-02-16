"""Continuation sampler emitter for multi-turn QD.

Generates follow-up user messages that continue existing conversations,
conditioned on parent conversations from the previous turn's archive.
"""

import asyncio
import random
from enum import Enum
from pathlib import Path

from flashlite import Flashlite
from jinja2 import Environment, FileSystemLoader, select_autoescape

from behavior_qd.archive import ContinuationArchive, PromptEntry
from behavior_qd.config import SamplerConfig
from behavior_qd.conversation import Conversation
from behavior_qd.emitters.base import BaseEmitter, EmitterFeedback, EmitterResult


class ContinuationStrategy(str, Enum):
    """Strategies for generating conversation continuations."""

    STANDARD = "standard"
    ESCALATION = "escalation"
    DEEPENING = "deepening"
    PIVOTING = "pivoting"
    CHALLENGING = "challenging"
    EMOTIONAL_APPEAL = "emotional_appeal"
    COMMITMENT_SEEKING = "commitment_seeking"
    MUTATION = "mutation"
    ANTI_ELITE = "anti_elite"


# Default strategy rotation for continuations
DEFAULT_CONTINUATION_STRATEGIES = [
    ContinuationStrategy.STANDARD,
    ContinuationStrategy.ESCALATION,
    ContinuationStrategy.DEEPENING,
    ContinuationStrategy.PIVOTING,
    ContinuationStrategy.CHALLENGING,
    ContinuationStrategy.EMOTIONAL_APPEAL,
    ContinuationStrategy.COMMITMENT_SEEKING,
]


class ContinuationSamplerEmitter(BaseEmitter):
    """Emitter that generates follow-up messages for existing conversations.

    Picks a parent conversation from the previous turn's archive, then
    uses an LLM to generate diverse follow-up messages. Supports strategy
    rotation, elite conditioning, and mutation — all adapted for the
    continuation context.
    """

    def __init__(
        self,
        archive: ContinuationArchive,
        config: SamplerConfig,
        behavior_description: str,
        parent_entries: list[PromptEntry],
        client: Flashlite | None = None,
        template_dir: Path | None = None,
    ):
        """Initialize the continuation sampler emitter.

        Args:
            archive: The continuation archive for this turn.
            config: Sampler configuration (reuses batch_size, model, etc.).
            behavior_description: Description of behavior to elicit.
            parent_entries: List of parent conversation entries from previous turn.
            client: Optional pre-configured flashlite client.
            template_dir: Directory containing Jinja templates.
        """
        # BaseEmitter expects BehaviorArchive but we use ContinuationArchive;
        # store directly to avoid type issues
        self._continuation_archive = archive
        self.config = config
        self.behavior_description = behavior_description
        self.parent_entries = parent_entries
        self._client = client
        self._template_dir = template_dir or Path(__file__).parent.parent / "templates"
        self._jinja_env: Environment | None = None

        # Strategy rotation state
        self._iteration_count = 0
        self._strategy_index = 0

        # Parent rotation — round-robin through parents
        self._parent_index = 0

        # Continuation-specific strategies
        self._strategies = list(DEFAULT_CONTINUATION_STRATEGIES)

        # Mutation types adapted for continuations
        self._mutation_types = [
            "change_tone",
            "change_angle",
            "add_context",
            "simplify",
            "escalate",
            "redirect",
        ]

    @property
    def archive(self) -> ContinuationArchive:
        """The continuation archive."""
        return self._continuation_archive

    @property
    def client(self) -> Flashlite:
        """Lazily create the flashlite client."""
        if self._client is None:
            self._client = Flashlite()
        return self._client

    @property
    def jinja_env(self) -> Environment:
        """Lazily create the Jinja environment."""
        if self._jinja_env is None:
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self._template_dir)),
                autoescape=select_autoescape(),
            )
        return self._jinja_env

    @property
    def batch_size(self) -> int:
        """Number of continuations generated per ask() call."""
        return self.config.batch_size

    def _select_parent(self) -> tuple[int, PromptEntry]:
        """Select the next parent conversation to extend.

        Uses round-robin to ensure all parents get continuations.

        Returns:
            Tuple of (parent_index, parent_entry).
        """
        idx = self._parent_index % len(self.parent_entries)
        self._parent_index += 1
        return idx, self.parent_entries[idx]

    def _select_strategy(self) -> ContinuationStrategy:
        """Select the continuation strategy for this iteration.

        Returns:
            The selected ContinuationStrategy.
        """
        # Check for anti-elite exploration
        num_elites = self.archive.stats["num_elites"]
        if (
            self.config.anti_elite_probability > 0
            and random.random() < self.config.anti_elite_probability
            and num_elites > 0
        ):
            return ContinuationStrategy.ANTI_ELITE

        # Check for mutation mode
        if (
            self.config.mutation_probability > 0
            and random.random() < self.config.mutation_probability
            and num_elites > 0
        ):
            return ContinuationStrategy.MUTATION

        # Strategy rotation
        strategy = self._strategies[self._strategy_index % len(self._strategies)]
        return strategy

    def _get_conversation_for_parent(self, parent: PromptEntry) -> Conversation:
        """Build the conversation object from a parent entry.

        Args:
            parent: The parent PromptEntry.

        Returns:
            Conversation representing the parent exchange.
        """
        if parent.conversation is not None:
            return parent.conversation
        # Turn 0 parent — build from prompt + response
        return Conversation.from_prompt_response(parent.prompt, parent.response)

    def _get_elite_continuations(self, parent_idx: int) -> list[str]:
        """Get elite continuation examples for a specific parent.

        Args:
            parent_idx: Index of the parent conversation.

        Returns:
            List of elite continuation texts.
        """
        elites = self.archive.get_elites_for_parent(parent_idx)
        if not elites:
            # Fall back to global elites if no parent-specific ones
            elites = self.archive.sample_elites(self.config.num_elite_examples)
        else:
            elites = elites[: self.config.num_elite_examples]
        return [e.prompt for e in elites]

    def _render_continuation_prompt(
        self,
        conversation: Conversation,
        parent_idx: int,
        strategy: ContinuationStrategy,
        domain: str | None = None,
    ) -> str:
        """Render the continuation sampler prompt.

        Args:
            conversation: The conversation to continue.
            parent_idx: Parent conversation index (for elite lookup).
            strategy: The continuation strategy to use.
            domain: Optional domain constraint.

        Returns:
            Rendered prompt string.
        """
        template = self.jinja_env.get_template("continuation_sampler.jinja")

        elite_examples = None
        needs_elites = (
            self.config.use_elite_examples
            or strategy == ContinuationStrategy.MUTATION
            or strategy == ContinuationStrategy.ANTI_ELITE
        )
        if needs_elites and self.config.num_elite_examples > 0:
            elite_examples = self._get_elite_continuations(parent_idx)

        return template.render(
            behavior_description=self.behavior_description,
            conversation_turns=conversation.messages,
            elite_examples=elite_examples or [],
            num_prompts=self.config.batch_size,
            strategy=strategy.value,
            domain=domain,
            mutation_types=self._mutation_types,
        )

    def ask(self) -> EmitterResult:
        """Generate a batch of continuation follow-up messages.

        Returns:
            EmitterResult with generated follow-ups and parent metadata.
        """
        return asyncio.run(self._ask_async())

    async def _ask_async(self) -> EmitterResult:
        """Async implementation of ask()."""
        # Select parent conversation and strategy
        parent_idx, parent_entry = self._select_parent()
        strategy = self._select_strategy()

        # Select domain if configured
        domain = None
        if self.config.use_domain_injection and self.config.domains:
            domain = random.choice(self.config.domains)

        # Build conversation from parent
        conversation = self._get_conversation_for_parent(parent_entry)

        # Render the continuation prompt
        sampler_prompt = self._render_continuation_prompt(
            conversation=conversation,
            parent_idx=parent_idx,
            strategy=strategy,
            domain=domain,
        )

        # Generate follow-ups using the sampler model
        response = await self.client.complete(
            model=self.config.model,
            messages=[{"role": "user", "content": sampler_prompt}],
            temperature=1.0,
        )

        # Parse follow-ups
        followups = self._parse_prompts(response.content)

        # Pad or trim to batch_size
        if len(followups) < self.config.batch_size:
            while len(followups) < self.config.batch_size:
                followups.append(followups[len(followups) % max(1, len(followups))])
        elif len(followups) > self.config.batch_size:
            followups = followups[: self.config.batch_size]

        # Store parent info in result metadata for the scheduler to use
        result = EmitterResult(prompts=followups)
        # Attach parent metadata — the scheduler uses this to build conversations
        result._parent_idx = parent_idx  # type: ignore[attr-defined]
        result._parent_entry = parent_entry  # type: ignore[attr-defined]
        result._conversation = conversation  # type: ignore[attr-defined]

        return result

    def _parse_prompts(self, response: str) -> list[str]:
        """Parse follow-up messages from the LLM response.

        Reuses the same parsing logic as SamplerEmitter.

        Args:
            response: Raw response from the sampler model.

        Returns:
            List of extracted follow-up messages.
        """
        lines = response.strip().split("\n")
        prompts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes
            for prefix in [".", ")", ":", "-", "*"]:
                if len(line) > 2 and line[0].isdigit() and line[1] == prefix:
                    line = line[2:].strip()
                    break
                elif (
                    len(line) > 3
                    and line[0].isdigit()
                    and line[1].isdigit()
                    and line[2] == prefix
                ):
                    line = line[3:].strip()
                    break

            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()

            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1]

            if line:
                prompts.append(line)

        return prompts

    def tell(self, feedback: EmitterFeedback) -> None:
        """Update emitter state based on feedback.

        Args:
            feedback: Feedback from evaluating the generated continuations.
        """
        self._iteration_count += 1
        self._strategy_index = (self._strategy_index + 1) % len(self._strategies)
