"""Evaluation pipeline for behavior-qd framework.

Handles target model interaction and judge model scoring using flashlite.
Supports both single-turn and multi-turn conversation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flashlite import Flashlite, RateLimitConfig
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pydantic import BaseModel, Field

from behavior_qd.config import BehaviorQDConfig
from behavior_qd.conversation import Conversation


class JudgeResponse(BaseModel):
    """Structured response from the judge model."""

    behavior_score: float = Field(
        description="How well the response exhibits the target behavior (1-10, or 0 for refusal)",
    )
    coherence_score: float | None = Field(
        default=None,
        description="Grammaticality/coherence of the prompt (1-10)",
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

    async def get_target_response(
        self,
        prompt: str | None = None,
        *,
        conversation: Conversation | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Get a response from the target model.

        Supports both single-turn (prompt string) and multi-turn
        (Conversation object) interactions.

        Args:
            prompt: The prompt to send (single-turn). Ignored if conversation is set.
            conversation: Full conversation for multi-turn. Takes precedence over prompt.
            max_tokens: Override max tokens (e.g. for Turn 1+ higher limits).

        Returns:
            The target model's response.
        """
        if conversation is not None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                *conversation.messages,
            ]
        else:
            messages = [
                # TODO: make system prompt evolvable
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

        response = await self.client.complete(
            model=self.config.target.model,
            messages=messages,
            max_tokens=max_tokens or self.config.target.max_tokens,
            temperature=self.config.target.temperature,
        )
        return response.content

    def _render_rubric_content(self, rubric_path: Path | None = None) -> str | None:
        """Render a rubric file to a content string for injection into judge templates.

        Rubric files contain only scoring guidelines (definition, levels,
        examples, edge cases).  They are rendered once here, then passed
        as ``rubric_content`` to the judge template which handles the
        interaction rendering and score-request framing.

        Args:
            rubric_path: Path to the rubric Jinja template.

        Returns:
            Rendered rubric string, or ``None`` if no rubric is configured.
        """
        rubric_path = rubric_path or self.config.judge.rubric_path
        if rubric_path is None or not rubric_path.exists():
            return None

        template_content = rubric_path.read_text()
        template = Template(template_content)
        # Rubrics may optionally use score_coherence to note coherence
        return template.render(
            score_coherence=self.config.judge.score_coherence,
        )

    def _render_judge_prompt(
        self,
        *,
        prompt: str | None = None,
        response: str | None = None,
        conversation: Conversation | None = None,
        behavior_description: str | None = None,
        rubric_content: str | None = None,
    ) -> str:
        """Render the appropriate judge template (single- or multi-turn).

        Args:
            prompt: User prompt (single-turn).
            response: Assistant response (single-turn).
            conversation: Full conversation (multi-turn).
            behavior_description: Description of behavior to evaluate.
            rubric_content: Pre-rendered rubric scoring guidelines, or None.

        Returns:
            Rendered judge prompt string.
        """
        behavior_description = behavior_description or self.config.behavior_description
        is_multi_turn = conversation is not None and conversation.num_turns > 1

        common_vars = {
            "behavior_description": behavior_description,
            "score_coherence": self.config.judge.score_coherence,
            "rubric_content": rubric_content or "",
        }

        if is_multi_turn:
            return self.render_template(
                "judge_conversation.jinja",
                conversation_turns=conversation.messages,
                **common_vars,
            )
        else:
            return self.render_template(
                "judge.jinja",
                prompt=prompt,
                response=response,
                **common_vars,
            )

    async def judge_interaction(
        self,
        prompt: str,
        response: str,
        behavior_description: str | None = None,
        rubric_path: Path | None = None,
        conversation: Conversation | None = None,
    ) -> JudgeResponse:
        """Judge an interaction between prompt and response.

        Supports both single-turn (prompt/response) and multi-turn
        (Conversation) evaluation. When a conversation is provided,
        the judge evaluates the full conversation thread.

        Args:
            prompt: The prompt that was sent (single-turn).
            response: The target model's response (single-turn).
            behavior_description: Description of behavior to evaluate.
            rubric_path: Path to custom rubric template.
            conversation: Full conversation for multi-turn judging.

        Returns:
            JudgeResponse with scores and reasoning.
        """
        rubric_content = self._render_rubric_content(rubric_path)

        judge_prompt = self._render_judge_prompt(
            prompt=prompt,
            response=response,
            conversation=conversation,
            behavior_description=behavior_description,
            rubric_content=rubric_content,
        )

        # Get structured response from judge
        result = await self.client.complete(
            model=self.config.judge.model,
            messages=[{"role": "user", "content": judge_prompt}],
            response_model=JudgeResponse,
        )

        return result

    async def evaluate(
        self,
        prompt: str,
        behavior_description: str | None = None,
        conversation: Conversation | None = None,
        max_tokens: int | None = None,
    ) -> EvaluationResult:
        """Evaluate a single prompt or conversation.

        Gets response from target model and judges the interaction.

        Args:
            prompt: The prompt to evaluate (single-turn).
            behavior_description: Description of behavior to evaluate.
            conversation: Full conversation for multi-turn evaluation.
            max_tokens: Override max tokens for target response.

        Returns:
            EvaluationResult with scores and details.
        """
        # Get target response
        response = await self.get_target_response(
            prompt, conversation=conversation, max_tokens=max_tokens
        )

        # For multi-turn, extend the conversation with the response
        full_conversation = None
        if conversation is not None:
            full_conversation = Conversation(
                turns=list(conversation.turns),
                parent_id=conversation.parent_id,
            )
            from behavior_qd.conversation import ConversationTurn

            full_conversation.turns.append(
                ConversationTurn(role="assistant", content=response)
            )

        # Judge the interaction
        judge_response = await self.judge_interaction(
            prompt=prompt,
            response=response,
            behavior_description=behavior_description,
            conversation=full_conversation,
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
        conversations: list[Conversation | None] | None = None,
        max_tokens: int | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of prompts or conversations.

        Args:
            prompts: List of prompts to evaluate (used for single-turn,
                or as labels for multi-turn).
            behavior_description: Description of behavior to evaluate.
            max_concurrency: Maximum concurrent evaluations.
            conversations: Optional list of Conversation objects for multi-turn.
                When provided, the full conversation is sent to the target model.
            max_tokens: Override max tokens for target responses.

        Returns:
            List of EvaluationResult objects.
        """
        behavior_description = behavior_description or self.config.behavior_description
        max_concurrency = max_concurrency or self.config.rate_limit.max_concurrency
        effective_max_tokens = max_tokens or self.config.target.max_tokens

        if conversations is None:
            conversations = [None] * len(prompts)

        # Build target requests — multi-turn sends full message history
        target_requests = []
        for prompt, conv in zip(prompts, conversations):
            if conv is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    *conv.messages,
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            target_requests.append(
                {
                    "model": self.config.target.model,
                    "messages": messages,
                    "max_tokens": effective_max_tokens,
                    "temperature": self.config.target.temperature,
                }
            )

        target_responses = await self.client.complete_many(
            target_requests,
            max_concurrency=max_concurrency,
        )

        # Check for empty responses
        empty_response_indices = set()
        for i, resp in enumerate(target_responses):
            if not resp.content:
                empty_response_indices.add(i)

        # Build judge prompts
        rubric_content = self._render_rubric_content()
        judge_prompts = []
        valid_indices = []

        for i, (prompt, conv, target_response) in enumerate(
            zip(prompts, conversations, target_responses)
        ):
            if i in empty_response_indices:
                continue

            valid_indices.append(i)

            # For multi-turn, build the complete conversation including the new response
            full_conv = None
            if conv is not None and conv.num_turns >= 1:
                from behavior_qd.conversation import ConversationTurn

                full_conv = Conversation(
                    turns=list(conv.turns)
                    + [
                        ConversationTurn(
                            role="assistant", content=target_response.content
                        )
                    ],
                    parent_id=conv.parent_id,
                )

            judge_prompt = self._render_judge_prompt(
                prompt=prompt,
                response=target_response.content,
                conversation=full_conv,
                behavior_description=behavior_description,
                rubric_content=rubric_content,
            )
            judge_prompts.append(judge_prompt)

        # Get all judge responses in parallel
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

        # Build results
        judge_response_map = dict(zip(valid_indices, judge_responses))

        results = []
        for i, (prompt, target_response) in enumerate(zip(prompts, target_responses)):
            if i in empty_response_indices:
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
        conversations: list[Conversation | None] | None = None,
        max_tokens: int | None = None,
    ) -> list[EvaluationResult]:
        """Synchronous version of evaluate_batch.

        Args:
            prompts: List of prompts to evaluate.
            behavior_description: Description of behavior to evaluate.
            max_concurrency: Maximum concurrent evaluations.
            conversations: Optional conversations for multi-turn.
            max_tokens: Override max tokens for target responses.

        Returns:
            List of EvaluationResult objects.
        """
        import asyncio

        return asyncio.run(
            self.evaluate_batch(
                prompts,
                behavior_description,
                max_concurrency,
                conversations=conversations,
                max_tokens=max_tokens,
            )
        )

    @property
    def total_cost(self) -> float:
        """Get total cost incurred so far."""
        return self.client.total_cost if hasattr(self.client, "total_cost") else 0.0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used so far."""
        return self.client.total_tokens if hasattr(self.client, "total_tokens") else 0
