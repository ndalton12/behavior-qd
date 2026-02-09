"""Sampler emitter using LLM generation with best-of-n selection."""

import asyncio
from pathlib import Path

from flashlite import Flashlite
from jinja2 import Environment, FileSystemLoader, select_autoescape

from behavior_qd.archive import BehaviorArchive
from behavior_qd.config import SamplerConfig
from behavior_qd.emitters.base import BaseEmitter, EmitterFeedback, EmitterResult


class SamplerEmitter(BaseEmitter):
    """Emitter that uses an LLM to generate candidate prompts.

    Optionally conditions on elite prompts from the archive (few-shot).
    Uses best-of-n selection based on judge scores.
    """

    def __init__(
        self,
        archive: BehaviorArchive,
        config: SamplerConfig,
        behavior_description: str,
        client: Flashlite | None = None,
        template_dir: Path | None = None,
    ):
        """Initialize the sampler emitter.

        Args:
            archive: The behavior archive instance.
            config: Sampler configuration.
            behavior_description: Description of behavior to elicit.
            client: Optional pre-configured flashlite client.
            template_dir: Directory containing Jinja templates.
        """
        super().__init__(archive)
        self.config = config
        self.behavior_description = behavior_description
        self._client = client
        self._template_dir = template_dir or Path(__file__).parent.parent / "templates"
        self._jinja_env: Environment | None = None

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
        """Number of prompts generated per ask() call."""
        return self.config.batch_size

    def _render_sampler_prompt(self, elite_examples: list[str] | None = None) -> str:
        """Render the sampler prompt template.

        Args:
            elite_examples: Optional list of elite prompts for few-shot.

        Returns:
            Rendered prompt string.
        """
        template = self.jinja_env.get_template("sampler.jinja")
        return template.render(
            behavior_description=self.behavior_description,
            elite_examples=elite_examples or [],
            num_prompts=self.config.batch_size,
        )

    def ask(self) -> EmitterResult:
        """Generate a batch of candidate prompts using the sampler model.

        Returns:
            EmitterResult containing generated prompts.
        """
        return asyncio.run(self._ask_async())

    async def _ask_async(self) -> EmitterResult:
        """Async implementation of ask()."""
        # Get elite examples if configured
        elite_examples = None
        if self.config.use_elite_examples and self.config.num_elite_examples > 0:
            elites = self.archive.sample_elites(self.config.num_elite_examples)
            if elites:
                elite_examples = [e.prompt for e in elites]

        # Render the sampler prompt
        sampler_prompt = self._render_sampler_prompt(elite_examples)

        # Generate prompts using the sampler model
        response = await self.client.complete(
            model=self.config.model,
            messages=[{"role": "user", "content": sampler_prompt}],
            temperature=1.0,  # High temperature for diversity
        )

        # Parse the response to extract prompts
        prompts = self._parse_prompts(response.content)

        # Ensure we have the right number of prompts
        if len(prompts) < self.config.batch_size:
            # Pad with variations if needed
            while len(prompts) < self.config.batch_size:
                prompts.append(prompts[len(prompts) % max(1, len(prompts))])
        elif len(prompts) > self.config.batch_size:
            prompts = prompts[: self.config.batch_size]

        return EmitterResult(prompts=prompts)

    def _parse_prompts(self, response: str) -> list[str]:
        """Parse prompts from the sampler model's response.

        Expects prompts to be numbered or on separate lines.

        Args:
            response: Raw response from the sampler model.

        Returns:
            List of extracted prompts.
        """
        lines = response.strip().split("\n")
        prompts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes like "1.", "1)", "-", "*"
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

            # Remove leading dash or asterisk
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()

            # Remove quotes if present
            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1]

            if line:
                prompts.append(line)

        return prompts

    def tell(self, feedback: EmitterFeedback) -> None:
        """Update emitter state based on feedback.

        For the sampler emitter, this is a no-op since the LLM doesn't
        have persistent state. The archive handles elite selection.

        Args:
            feedback: Feedback from evaluating the generated prompts.
        """
        # Sampler emitter is stateless - archive handles everything
        pass

    async def ask_with_selection(
        self,
        evaluator,
        max_concurrency: int = 10,
    ) -> tuple[EmitterResult, list]:
        """Generate prompts and return with preliminary scores.

        This is useful for best-of-n selection before full evaluation.

        Args:
            evaluator: Evaluator instance for scoring.
            max_concurrency: Max concurrent evaluations.

        Returns:
            Tuple of (EmitterResult, list of EvaluationResults)
        """
        result = await self._ask_async()

        # Evaluate all generated prompts
        eval_results = await evaluator.evaluate_batch(
            result.prompts,
            max_concurrency=max_concurrency,
        )

        # Sort by score and keep best_of_n
        sorted_results = sorted(
            zip(result.prompts, eval_results),
            key=lambda x: x[1].final_score,
            reverse=True,
        )

        best_prompts = [p for p, _ in sorted_results[: self.config.best_of_n]]
        best_results = [r for _, r in sorted_results[: self.config.best_of_n]]

        return EmitterResult(prompts=best_prompts), best_results
