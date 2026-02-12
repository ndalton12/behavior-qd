"""Configuration settings for the behavior-qd framework."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmitterMode(str, Enum):
    """Which emitter strategy to use."""

    SAMPLER = "sampler"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding system."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    pca_components: int = Field(
        default=3,
        ge=3,
        description="Number of PCA components (archive uses 2nd and 3rd dimensions)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of nearest tokens to consider when snapping",
    )
    temperature: float = Field(
        default=0.5,
        ge=0.0,
        description="Temperature for token snapping (0 = deterministic argmax)",
    )
    max_prompt_length: int = Field(
        default=20,
        ge=3,
        description="Maximum prompt length in tokens for embedding emitter",
    )
    device: str | None = Field(
        default=None,
        description="Device to use for embedding model",
    )


class ArchiveConfig(BaseModel):
    """Configuration for the QD archive."""

    num_cells: int = Field(
        default=1000,
        ge=100,
        description="Number of CVT cells in the archive",
    )
    pca_range: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Range for PCA measure dimensions (normalized)",
    )
    distance_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Range for token distance-traveled measure dimension",
    )
    seed: int = Field(
        default=42,
        description="Random seed for CVT centroid initialization",
    )


class JudgeConfig(BaseModel):
    """Configuration for the judge model evaluation."""

    model: str = Field(
        default="gpt-5-mini",
        description="Model to use for judging behavior elicitation",
    )
    score_coherence: bool = Field(
        default=False,
        description="Whether to also score prompt coherence",
    )
    coherence_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight for coherence score in final objective (0 = ignore)",
    )
    rubric_path: Path | None = Field(
        default=None,
        description="Path to custom rubric Jinja template",
    )


class TargetConfig(BaseModel):
    """Configuration for the target model."""

    model: str = Field(
        default="gpt-5-mini",
        description="Model to elicit behaviors from",
    )
    max_tokens: int = Field(
        default=256,
        description="Maximum tokens in target model response",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        description="Temperature for target model generation",
    )


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting."""

    requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute",
    )
    tokens_per_minute: int = Field(
        default=100_000,
        ge=1,
        description="Maximum tokens per minute",
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent API requests",
    )


class SamplingStrategy(str, Enum):
    """Different strategies for generating prompts."""

    STANDARD = "standard"  # Default generation with elite examples
    ROLEPLAY = "roleplay"  # Generate prompts that ask AI to play a character
    HYPOTHETICAL = "hypothetical"  # Frame prompts as thought experiments
    PERSONAL = "personal"  # User shares a personal story/situation
    SOCRATIC = "socratic"  # Questions that probe through questioning
    INDIRECT = "indirect"  # Third-person, advice-seeking, or abstract framing
    EDGE_CASE = "edge_case"  # Extreme, boundary-pushing scenarios
    MUTATION = "mutation"  # Mutate existing elite prompts
    ANTI_ELITE = "anti_elite"  # Generate prompts maximally different from elites


# Default domains for diversity injection
DEFAULT_DOMAINS: list[str] = [
    "workplace and professional life",
    "family and personal relationships",
    "health and medical decisions",
    "financial and economic choices",
    "political and social issues",
    "technology and AI",
    "education and learning",
    "creative and artistic pursuits",
    "ethics and moral dilemmas",
    "science and research",
    "legal and regulatory matters",
    "environmental and sustainability",
]


class SamplerConfig(BaseModel):
    """Configuration for the sampler emitter."""

    model: str = Field(
        default="gpt-5-mini",
        description="Model to use for generating candidate prompts",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of prompts to generate per iteration",
    )
    best_of_n: int = Field(
        default=5,
        ge=1,
        description="Number of top prompts to keep from each batch",
    )
    use_elite_examples: bool = Field(
        default=True,
        description="Whether to condition generation on elite prompts from archive",
    )
    num_elite_examples: int = Field(
        default=3,
        ge=0,
        description="Number of elite examples to include in sampler prompt",
    )

    # Strategy settings
    strategies: list[SamplingStrategy] = Field(
        default=[
            SamplingStrategy.STANDARD,
            SamplingStrategy.ROLEPLAY,
            SamplingStrategy.HYPOTHETICAL,
            SamplingStrategy.PERSONAL,
            SamplingStrategy.SOCRATIC,
            SamplingStrategy.INDIRECT,
            SamplingStrategy.EDGE_CASE,
        ],
        description="List of strategies to rotate through",
    )
    strategy_rotation: bool = Field(
        default=True,
        description="Whether to rotate through strategies each iteration",
    )

    # Domain injection
    use_domain_injection: bool = Field(
        default=True,
        description="Whether to inject random domains for diversity",
    )
    domains: list[str] = Field(
        default_factory=lambda: DEFAULT_DOMAINS.copy(),
        description="List of domains to inject for topical diversity",
    )

    # Mutation settings
    mutation_probability: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of using mutation mode instead of strategy",
    )
    mutation_types: list[str] = Field(
        default=[
            "change_domain",
            "change_framing",
            "change_tone",
            "add_context",
            "simplify",
            "make_indirect",
            "escalate",
        ],
        description="Types of mutations to apply",
    )

    # Anti-elite exploration
    anti_elite_probability: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Probability of using anti-elite exploration mode",
    )


class EmbeddingEmitterConfig(BaseModel):
    """Configuration for the CMA-ME embedding emitter."""

    batch_size: int = Field(
        default=36,
        ge=1,
        description="Number of solutions per CMA-ME iteration",
    )
    sigma0: float = Field(
        default=0.1,
        gt=0.0,
        description="Initial standard deviation for CMA-ES",
    )
    num_emitters: int = Field(
        default=3,
        ge=1,
        description="Number of parallel CMA-ME emitters",
    )


class SchedulerConfig(BaseModel):
    """Configuration for the QD scheduler."""

    emitter_mode: EmitterMode = Field(
        default=EmitterMode.SAMPLER,
        description="Which emitter strategy to use",
    )
    iterations: int = Field(
        default=100,
        ge=1,
        description="Number of QD iterations to run",
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N iterations",
    )
    log_interval: int = Field(
        default=1,
        ge=1,
        description="Log progress every N iterations",
    )
    seed_file: Path | None = Field(
        default=None,
        description="CSV file with seed prompts (must have 'prompt' column)",
    )


class BehaviorQDConfig(BaseSettings):
    """Main configuration for the behavior-qd framework."""

    model_config = SettingsConfigDict(
        env_prefix="BEHAVIOR_QD_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Behavior to elicit
    behavior_description: str = Field(
        default="",
        description="Description of the behavior to elicit (e.g., 'sycophancy')",
    )

    # Output paths
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for outputs, checkpoints, and logs",
    )
    template_dir: Path = Field(
        default=Path(__file__).parent / "templates",
        description="Directory containing Jinja templates",
    )

    # Sub-configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    embedding_emitter: EmbeddingEmitterConfig = Field(
        default_factory=EmbeddingEmitterConfig
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    def get_output_path(self, filename: str) -> Path:
        """Get path to an output file, creating directory if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename
