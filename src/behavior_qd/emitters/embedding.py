"""Embedding emitter using CMA-ME with token snapping."""

import numpy as np
from numpy.typing import NDArray
from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from scipy.cluster.vq import kmeans

from behavior_qd.archive import BehaviorArchive
from behavior_qd.config import EmbeddingConfig, EmbeddingEmitterConfig
from behavior_qd.embeddings import EmbeddingSpace
from behavior_qd.emitters.base import BaseEmitter, EmitterFeedback, EmitterResult


class EmbeddingEmitter(BaseEmitter):
    """Emitter that uses CMA-ME to explore embedding space.

    Evolves sequences of embedding vectors and snaps them to discrete
    tokens using temperature-based top-k sampling.
    """

    def __init__(
        self,
        archive: BehaviorArchive,
        embedding_space: EmbeddingSpace,
        emitter_config: EmbeddingEmitterConfig,
        embedding_config: EmbeddingConfig,
    ):
        """Initialize the embedding emitter.

        Args:
            archive: The behavior archive instance.
            embedding_space: Embedding space for snapping.
            emitter_config: Emitter-specific configuration.
            embedding_config: Embedding configuration (for snapping params).
        """
        super().__init__(archive)
        self.embedding_space = embedding_space
        self.emitter_config = emitter_config
        self.embedding_config = embedding_config

        # Calculate genome size
        self._genome_size = embedding_space.genome_size(
            embedding_config.max_prompt_length
        )

        # Create internal archive for CMA-ME (separate from BehaviorArchive)
        # This archive stores full genomes, while BehaviorArchive stores prompt IDs
        self._internal_archive = self._create_internal_archive()

        # Initialize CMA-ME emitters
        self._emitters = self._create_emitters()

        # Track current batch for tell()
        self._current_genomes: NDArray[np.float32] | None = None

    def _create_internal_archive(self) -> CVTArchive:
        """Create internal CVTArchive for CMA-ME with genome-sized solutions.

        Returns:
            CVTArchive configured for genome storage.
        """
        # Use same ranges as the main archive (2D: dim1, dim2)
        ranges = [
            self.archive.config.dim_range,  # dim1
            self.archive.config.dim_range,  # dim2
        ]

        # Generate centroids
        rng = np.random.default_rng(self.archive.config.seed + 1000)  # Different seed
        num_cells = self.archive.config.num_cells
        num_samples = num_cells * 100
        samples = rng.uniform(
            low=[r[0] for r in ranges],
            high=[r[1] for r in ranges],
            size=(num_samples, len(ranges)),
        )
        centroids, _ = kmeans(samples, num_cells)

        return CVTArchive(
            solution_dim=self._genome_size,
            ranges=ranges,
            centroids=centroids,
        )

    def _create_emitters(self) -> list[EvolutionStrategyEmitter]:
        """Create the underlying pyribs emitters.

        Returns:
            List of EvolutionStrategyEmitter instances.
        """
        emitters = []

        for i in range(self.emitter_config.num_emitters):
            # Initialize with random embedding sequence
            x0 = self._random_genome(seed=i)

            emitter = EvolutionStrategyEmitter(
                archive=self._internal_archive,  # Use internal archive with genome-sized solutions
                x0=x0,
                sigma0=self.emitter_config.sigma0,
                batch_size=self.emitter_config.batch_size
                // self.emitter_config.num_emitters,
            )
            emitters.append(emitter)

        return emitters

    def _random_genome(self, seed: int | None = None) -> NDArray[np.float32]:
        """Generate a random genome for initialization.

        Args:
            seed: Random seed.

        Returns:
            Random genome array.
        """
        if seed is not None:
            np.random.seed(seed)

        max_len = self.embedding_config.max_prompt_length
        embed_dim = self.embedding_space.embed_dim

        # Random embedding sequence
        embeddings = self.embedding_space.random_embedding_sequence(max_len, seed=seed)

        # Random length parameter (favor medium lengths)
        length_param = np.random.beta(2, 2)  # Peaks around 0.5

        # Flatten and concatenate
        genome = np.concatenate(
            [
                embeddings.flatten(),
                [length_param],
            ]
        ).astype(np.float32)

        return genome

    def _genome_to_prompt(
        self,
        genome: NDArray[np.float32],
        seed: int | None = None,
    ) -> str:
        """Convert a genome to a prompt string.

        Args:
            genome: Flattened genome array.
            seed: Random seed for token snapping.

        Returns:
            Decoded prompt string.
        """
        # Extract embedding sequence and length
        embeddings, length = self.embedding_space.embedding_sequence_with_length(
            genome,
            max_length=self.embedding_config.max_prompt_length,
        )

        # Snap to tokens
        prompt = self.embedding_space.snap_to_tokens(
            embeddings,
            top_k=self.embedding_config.top_k,
            temperature=self.embedding_config.temperature,
            seed=seed,
        )

        return prompt

    @property
    def batch_size(self) -> int:
        """Number of prompts generated per ask() call."""
        return self.emitter_config.batch_size

    def ask(self) -> EmitterResult:
        """Generate a batch of candidate prompts via CMA-ME.

        Returns:
            EmitterResult containing prompts and genomes.
        """
        all_genomes = []

        # Ask each emitter for solutions
        for emitter in self._emitters:
            genomes = emitter.ask()
            all_genomes.append(genomes)

        # Combine all genomes
        self._current_genomes = np.vstack(all_genomes)

        # Convert genomes to prompts
        prompts = []
        for i, genome in enumerate(self._current_genomes):
            # Use index as seed for reproducibility within batch
            prompt = self._genome_to_prompt(genome, seed=i)
            prompts.append(prompt)

        return EmitterResult(
            prompts=prompts,
            genomes=self._current_genomes,
        )

    def tell(self, feedback: EmitterFeedback) -> None:
        """Update emitter state based on evaluation feedback.

        Args:
            feedback: Feedback from evaluating the generated prompts.
        """
        if self._current_genomes is None:
            raise ValueError("tell() called before ask()")

        # Convert measures to array
        measures = np.array([m.as_array() for m in feedback.measures])
        objectives = np.array(feedback.objectives)

        # Add solutions to internal archive and get add_info for emitters
        # pyribs 0.9+ requires solution and add_info in tell()
        add_info = self._internal_archive.add(
            solution=self._current_genomes,
            objective=objectives,
            measures=measures,
        )

        # Split feedback among emitters
        batch_per_emitter = (
            self.emitter_config.batch_size // self.emitter_config.num_emitters
        )

        for i, emitter in enumerate(self._emitters):
            start = i * batch_per_emitter
            end = start + batch_per_emitter

            # Slice add_info for this emitter's batch
            emitter_add_info = {key: val[start:end] for key, val in add_info.items()}

            emitter.tell(
                solution=self._current_genomes[start:end],
                objective=objectives[start:end],
                measures=measures[start:end],
                add_info=emitter_add_info,
            )

        # Clear current genomes
        self._current_genomes = None

    def tell_with_solutions(
        self,
        solutions: NDArray[np.float32],
        objectives: NDArray[np.float32],
        measures: NDArray[np.float32],
    ) -> None:
        """Tell emitters with explicit solutions (for archive integration).

        This is useful when the archive handles solution storage differently.

        Args:
            solutions: Solution genomes.
            objectives: Objective scores.
            measures: Measure values.
        """
        # Add to internal archive to get add_info
        add_info = self._internal_archive.add(
            solution=solutions,
            objective=objectives,
            measures=measures,
        )

        batch_per_emitter = len(solutions) // self.emitter_config.num_emitters

        for i, emitter in enumerate(self._emitters):
            start = i * batch_per_emitter
            end = start + batch_per_emitter

            emitter_add_info = {key: val[start:end] for key, val in add_info.items()}

            emitter.tell(
                solution=solutions[start:end],
                objective=objectives[start:end],
                measures=measures[start:end],
                add_info=emitter_add_info,
            )


class HybridEmitter(BaseEmitter):
    """Combines sampler and embedding emitters.

    Alternates between or runs both emitters in parallel.
    """

    def __init__(
        self,
        archive: BehaviorArchive,
        sampler_emitter: "SamplerEmitter",  # noqa: F821 - forward ref
        embedding_emitter: EmbeddingEmitter,
        sampler_ratio: float = 0.5,
    ):
        """Initialize the hybrid emitter.

        Args:
            archive: The behavior archive instance.
            sampler_emitter: Sampler emitter instance.
            embedding_emitter: Embedding emitter instance.
            sampler_ratio: Ratio of prompts from sampler (0-1).
        """
        super().__init__(archive)
        self.sampler_emitter = sampler_emitter
        self.embedding_emitter = embedding_emitter
        self.sampler_ratio = sampler_ratio

        self._last_sampler_result: EmitterResult | None = None
        self._last_embedding_result: EmitterResult | None = None

    @property
    def batch_size(self) -> int:
        """Combined batch size from both emitters."""
        return self.sampler_emitter.batch_size + self.embedding_emitter.batch_size

    def ask(self) -> EmitterResult:
        """Generate prompts from both emitters.

        Returns:
            Combined EmitterResult.
        """
        # Get prompts from both emitters
        self._last_sampler_result = self.sampler_emitter.ask()
        self._last_embedding_result = self.embedding_emitter.ask()

        # Combine prompts
        all_prompts = (
            self._last_sampler_result.prompts + self._last_embedding_result.prompts
        )

        return EmitterResult(
            prompts=all_prompts,
            genomes=self._last_embedding_result.genomes,
        )

    def tell(self, feedback: EmitterFeedback) -> None:
        """Update both emitters based on feedback.

        Args:
            feedback: Feedback from evaluating all generated prompts.
        """
        sampler_size = self.sampler_emitter.batch_size
        embedding_size = self.embedding_emitter.batch_size

        # Split feedback
        sampler_feedback = EmitterFeedback(
            prompts=feedback.prompts[:sampler_size],
            objectives=feedback.objectives[:sampler_size],
            measures=feedback.measures[:sampler_size],
            statuses=feedback.statuses[:sampler_size] if feedback.statuses else None,
        )

        embedding_feedback = EmitterFeedback(
            prompts=feedback.prompts[sampler_size:],
            objectives=feedback.objectives[sampler_size:],
            measures=feedback.measures[sampler_size:],
            statuses=feedback.statuses[sampler_size:] if feedback.statuses else None,
        )

        # Tell both emitters
        self.sampler_emitter.tell(sampler_feedback)
        self.embedding_emitter.tell(embedding_feedback)
