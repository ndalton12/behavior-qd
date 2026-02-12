"""Evaluation pipeline for behavior-qd framework.

Handles target model interaction and judge model scoring using flashlite.
"""

from dataclasses import dataclass
from pathlib import Path

from flashlite import Flashlite, RateLimitConfig
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pydantic import BaseModel, Field

from behavior_qd.config import BehaviorQDConfig


class JudgeResponse(BaseModel):
    """Structured response from the judge model."""

    behavior_score: float = Field(
        description="How well the response exhibits the target behavior (scale defined by rubric)",
    )
    coherence_score: float | None = Field(
        default=None,
        description="Grammaticality/coherence of the prompt (scale defined by rubric)",
    )
    reasoning: str = Field(
        description="Brief explanation for the scores",
    )


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt against target and judge."""

    prompt: str
    response: str
    behavior_score: float
    coherence_score: float | None
    final_score: float
    reasoning: str

    @classmethod
    def from_judge_response(
        cls,
        prompt: str,
        response: str,
        judge_response: JudgeResponse,
        coherence_weight: float = 0.0,
    ) -> "EvaluationResult":
        """Create an EvaluationResult from a JudgeResponse."""
        # Compute final score with optional coherence weighting
        final_score = judge_response.behavior_score
        if judge_response.coherence_score is not None and coherence_weight > 0:
            final_score += coherence_weight * judge_response.coherence_score

        return cls(
            prompt=prompt,
            response=response,
            behavior_score=judge_response.behavior_score,
            coherence_score=judge_response.coherence_score,
            final_score=final_score,
            reasoning=judge_response.reasoning,
        )


class Evaluator:
    """Handles evaluation of prompts against target and judge models."""

    def __init__(
        self,
        config: BehaviorQDConfig,
        client: Flashlite | None = None,
    ):
        """Initialize the evaluator.

        Args:
            config: Main configuration object.
            client: Optional pre-configured flashlite client.
        """
        self.config = config
        self._client = client
        self._jinja_env: Environment | None = None

    @property
    def client(self) -> Flashlite:
        """Lazily create the flashlite client."""
        if self._client is None:
            rate_limit_cfg = self.config.rate_limit
            self._client = Flashlite(
                rate_limit=RateLimitConfig(
                    requests_per_minute=rate_limit_cfg.requests_per_minute,
                    tokens_per_minute=rate_limit_cfg.tokens_per_minute,
                ),
                track_costs=True,
            )
        return self._client

    @property
    def jinja_env(self) -> Environment:
        """Lazily create the Jinja environment."""
        if self._jinja_env is None:
            template_dir = self.config.template_dir
            if not template_dir.exists():
                # Fall back to package templates
                template_dir = Path(__file__).parent / "templates"

            self._jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(),
            )
        return self._jinja_env

    def render_template(self, template_name: str, **variables) -> str:
        """Render a Jinja template with variables.

        Args:
            template_name: Name of the template file (e.g., "judge.jinja")
            **variables: Variables to pass to the template.

        Returns:
            Rendered template string.
        """
        template = self.jinja_env.get_template(template_name)
        return template.render(**variables)

    async def get_target_response(self, prompt: str) -> str:
        """Get a response from the target model.

        Args:
            prompt: The prompt to send to the target model.

        Returns:
            The target model's response.
        """
        response = await self.client.complete(
            model=self.config.target.model,
            messages=[
                # TODO: make system prompt evolvable
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.target.max_tokens,
            temperature=self.config.target.temperature,
        )
        return response.content

    async def judge_interaction(
        self,
        prompt: str,
        response: str,
        behavior_description: str | None = None,
        rubric_path: Path | None = None,
    ) -> JudgeResponse:
        """Judge an interaction between prompt and response.

        Args:
            prompt: The prompt that was sent.
            response: The target model's response.
            behavior_description: Description of behavior to evaluate.
            rubric_path: Path to custom rubric template.

        Returns:
            JudgeResponse with scores and reasoning.
        """
        behavior_description = behavior_description or self.config.behavior_description
        rubric_path = rubric_path or self.config.judge.rubric_path

        # Determine which template to use
        if rubric_path and rubric_path.exists():
            # Use custom rubric
            template_content = rubric_path.read_text()

            template = Template(template_content)
            judge_prompt = template.render(
                prompt=prompt,
                response=response,
                behavior_description=behavior_description,
                score_coherence=self.config.judge.score_coherence,
            )
        else:
            # Use default judge template
            judge_prompt = self.render_template(
                "judge.jinja",
                prompt=prompt,
                response=response,
                behavior_description=behavior_description,
                score_coherence=self.config.judge.score_coherence,
            )

        # Get structured response from judge
        result = await self.client.complete(
            model=self.config.judge.model,
            messages=[{"role": "user", "content": judge_prompt}],
            response_model=JudgeResponse,
            # temperature=0.0,  # Deterministic for consistency
        )

        return result

    async def evaluate(
        self,
        prompt: str,
        behavior_description: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single prompt.

        Gets response from target model and judges the interaction.

        Args:
            prompt: The prompt to evaluate.
            behavior_description: Description of behavior to evaluate.

        Returns:
            EvaluationResult with scores and details.
        """
        # Get target response
        response = await self.get_target_response(prompt)

        # Judge the interaction
        judge_response = await self.judge_interaction(
            prompt=prompt,
            response=response,
            behavior_description=behavior_description,
        )

        return EvaluationResult.from_judge_response(
            prompt=prompt,
            response=response,
            judge_response=judge_response,
            coherence_weight=self.config.judge.coherence_weight,
        )

    async def evaluate_batch(
        self,
        prompts: list[str],
        behavior_description: str | None = None,
        max_concurrency: int | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of prompts.

        Args:
            prompts: List of prompts to evaluate.
            behavior_description: Description of behavior to evaluate.
            max_concurrency: Maximum concurrent evaluations. Defaults to config value.

        Returns:
            List of EvaluationResult objects.
        """
        behavior_description = behavior_description or self.config.behavior_description
        max_concurrency = max_concurrency or self.config.rate_limit.max_concurrency

        # First, get all target responses in parallel
        target_requests = [
            {
                "model": self.config.target.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.target.max_tokens,
                "temperature": self.config.target.temperature,
            }
            for prompt in prompts
        ]

        target_responses = await self.client.complete_many(
            target_requests,
            max_concurrency=max_concurrency,
        )

        # Check for empty responses (likely model refusals)
        empty_response_indices = set()
        for i, resp in enumerate(target_responses):
            if not resp.content:
                empty_response_indices.add(i)

        # Build judge prompts (only for non-empty responses)
        rubric_path = self.config.judge.rubric_path
        judge_prompts = []
        valid_indices = []

        for i, (prompt, target_response) in enumerate(zip(prompts, target_responses)):
            if i in empty_response_indices:
                continue  # Skip empty responses

            valid_indices.append(i)
            if rubric_path and rubric_path.exists():
                template_content = rubric_path.read_text()

                template = Template(template_content)
                judge_prompt = template.render(
                    prompt=prompt,
                    response=target_response.content,
                    behavior_description=behavior_description,
                    score_coherence=self.config.judge.score_coherence,
                )
            else:
                judge_prompt = self.render_template(
                    "judge.jinja",
                    prompt=prompt,
                    response=target_response.content,
                    behavior_description=behavior_description,
                    score_coherence=self.config.judge.score_coherence,
                )
            judge_prompts.append(judge_prompt)

        # Get all judge responses in parallel (only for valid responses)
        judge_requests = [
            {
                "model": self.config.judge.model,
                "messages": [{"role": "user", "content": jp}],
                "response_model": JudgeResponse,
            }
            for jp in judge_prompts
        ]

        if judge_requests:
            judge_responses = await self.client.complete_many(
                judge_requests,
                max_concurrency=max_concurrency,
            )
        else:
            judge_responses = []

        # Build results - map valid judge responses back to original indices
        judge_response_map = dict(zip(valid_indices, judge_responses))

        results = []
        for i, (prompt, target_response) in enumerate(zip(prompts, target_responses)):
            if i in empty_response_indices:
                # Empty response - create a zero-score result
                result = EvaluationResult(
                    prompt=prompt,
                    response="[MODEL REFUSED/EMPTY RESPONSE]",
                    behavior_score=0.0,
                    coherence_score=0.0 if self.config.judge.score_coherence else None,
                    final_score=0.0,
                    reasoning="Model returned empty response (likely refusal).",
                )
            else:
                judge_response = judge_response_map[i]
                result = EvaluationResult.from_judge_response(
                    prompt=prompt,
                    response=target_response.content,
                    judge_response=judge_response,
                    coherence_weight=self.config.judge.coherence_weight,
                )
            results.append(result)

        return results

    def evaluate_sync(
        self,
        prompt: str,
        behavior_description: str | None = None,
    ) -> EvaluationResult:
        """Synchronous version of evaluate.

        Args:
            prompt: The prompt to evaluate.
            behavior_description: Description of behavior to evaluate.

        Returns:
            EvaluationResult with scores and details.
        """
        import asyncio

        return asyncio.run(self.evaluate(prompt, behavior_description))

    def evaluate_batch_sync(
        self,
        prompts: list[str],
        behavior_description: str | None = None,
        max_concurrency: int | None = None,
    ) -> list[EvaluationResult]:
        """Synchronous version of evaluate_batch.

        Args:
            prompts: List of prompts to evaluate.
            behavior_description: Description of behavior to evaluate.
            max_concurrency: Maximum concurrent evaluations. Defaults to config value.

        Returns:
            List of EvaluationResult objects.
        """
        import asyncio

        return asyncio.run(
            self.evaluate_batch(prompts, behavior_description, max_concurrency)
        )

    @property
    def total_cost(self) -> float:
        """Get total cost incurred so far."""
        return self.client.total_cost if hasattr(self.client, "total_cost") else 0.0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used so far."""
        return self.client.total_tokens if hasattr(self.client, "total_tokens") else 0
