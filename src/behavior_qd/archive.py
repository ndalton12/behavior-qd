"""Archive management for behavior-qd framework.

Wraps pyribs CVTArchive with behavior-specific functionality.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from ribs.archives import CVTArchive, AddStatus
from scipy.cluster.vq import kmeans

from behavior_qd.config import ArchiveConfig
from behavior_qd.embeddings import EmbeddingSpace, Measures


@dataclass
class PromptEntry:
    """An entry in the archive representing an elicited prompt."""

    prompt: str
    objective: float
    measures: Measures
    response: str | None = None
    reasoning: str | None = None
    metadata: dict | None = None


class BehaviorArchive:
    """Archive for storing diverse behavior-eliciting prompts.

    Uses CVTArchive with 3D measures:
    - PCA dimension 1 (from vocab embedding PCA)
    - PCA dimension 2
    - Token embedding variance
    """

    def __init__(
        self,
        config: ArchiveConfig,
        embedding_space: EmbeddingSpace,
    ):
        """Initialize the behavior archive.

        Args:
            config: Archive configuration.
            embedding_space: Embedding space for measure computation.
        """
        self.config = config
        self.embedding_space = embedding_space

        # Define measure ranges
        ranges = [
            config.pca_range,  # PCA dim 1
            config.pca_range,  # PCA dim 2
            config.variance_range,  # Token variance
        ]

        # Generate centroids using k-means
        rng = np.random.default_rng(config.seed)
        num_samples = config.num_cells * 100  # Oversample for better k-means
        samples = rng.uniform(
            low=[r[0] for r in ranges],
            high=[r[1] for r in ranges],
            size=(num_samples, len(ranges)),
        )
        centroids, _ = kmeans(samples, config.num_cells)

        # Create CVTArchive with pre-computed centroids
        self._archive = CVTArchive(
            solution_dim=1,  # We store index into prompt_data
            ranges=ranges,
            centroids=centroids,
        )

        # Store actual prompt data separately (CVTArchive stores numpy arrays)
        self._prompt_data: dict[int, PromptEntry] = {}
        self._next_id = 0

    @property
    def archive(self) -> CVTArchive:
        """Access the underlying pyribs archive."""
        return self._archive

    def add(
        self,
        prompt: str,
        objective: float,
        measures: Measures | None = None,
        response: str | None = None,
        reasoning: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[AddStatus, int | None]:
        """Add a prompt to the archive.

        Args:
            prompt: The prompt text.
            objective: Objective score (higher is better).
            measures: Pre-computed measures, or None to compute.
            response: Optional target model response.
            reasoning: Optional judge reasoning.
            metadata: Optional additional metadata.

        Returns:
            Tuple of (AddStatus, cell_index or None if not added)
        """
        # Compute measures if not provided
        if measures is None:
            measures = self.embedding_space.compute_measures(prompt)

        # Create entry
        entry = PromptEntry(
            prompt=prompt,
            objective=objective,
            measures=measures,
            response=response,
            reasoning=reasoning,
            metadata=metadata,
        )

        # Assign ID and store
        entry_id = self._next_id
        self._next_id += 1

        # Add to pyribs archive
        # Solution is just the entry ID (we store actual data separately)
        result = self._archive.add_single(
            solution=np.array([entry_id]),
            objective=objective,
            measures=measures.as_array(),
        )

        # pyribs 0.9+ returns dict with 'status' key; status > 0 means added
        status = result["status"] if isinstance(result, dict) else result
        was_added = int(status) > 0  # NEW=1, IMPROVE=2

        if was_added:
            self._prompt_data[entry_id] = entry
            return status, self._archive.index_of_single(measures.as_array())
        else:
            return status, None

    def add_batch(
        self,
        prompts: list[str],
        objectives: list[float],
        measures_list: list[Measures] | None = None,
        responses: list[str | None] | None = None,
        reasonings: list[str | None] | None = None,
    ) -> list[AddStatus]:
        """Add a batch of prompts to the archive.

        Args:
            prompts: List of prompt texts.
            objectives: List of objective scores.
            measures_list: Optional pre-computed measures.
            responses: Optional target model responses.
            reasonings: Optional judge reasonings.

        Returns:
            List of AddStatus for each prompt.
        """
        n = len(prompts)

        # Compute measures if not provided
        if measures_list is None:
            measures_list = [
                self.embedding_space.compute_measures(p) for p in prompts
            ]

        # Default Nones
        if responses is None:
            responses = [None] * n
        if reasonings is None:
            reasonings = [None] * n

        # Prepare arrays for pyribs batch add
        solutions = []
        objs = []
        meas = []

        entry_ids = []
        entries = []

        for i, (prompt, obj, measures, response, reasoning) in enumerate(
            zip(prompts, objectives, measures_list, responses, reasonings)
        ):
            entry = PromptEntry(
                prompt=prompt,
                objective=obj,
                measures=measures,
                response=response,
                reasoning=reasoning,
            )
            entry_id = self._next_id + i
            entry_ids.append(entry_id)
            entries.append(entry)

            solutions.append([entry_id])
            objs.append(obj)
            meas.append(measures.as_array())

        self._next_id += n

        # Batch add to archive
        result = self._archive.add(
            solution=np.array(solutions),
            objective=np.array(objs),
            measures=np.array(meas),
        )

        # pyribs 0.9+ returns dict with 'status' key
        status_array = result["status"] if isinstance(result, dict) else result

        # Store entries that were added (status > 0 means NEW=1 or IMPROVE=2)
        for entry_id, entry, status in zip(entry_ids, entries, status_array):
            if int(status) > 0:
                self._prompt_data[entry_id] = entry

        return list(status_array)

    def get_elite(self, index: int) -> PromptEntry | None:
        """Get an elite entry by cell index.

        Args:
            index: Cell index in the archive.

        Returns:
            PromptEntry or None if cell is empty.
        """
        if not self._archive.occupied[index]:
            return None

        # Get the solution (entry ID) from the archive
        entry_id = int(self._archive.data("solution")[index][0])
        return self._prompt_data.get(entry_id)

    def get_elites(self, n: int | None = None) -> list[PromptEntry]:
        """Get top n elites by objective score.

        Args:
            n: Number of elites to return. None for all.

        Returns:
            List of PromptEntry sorted by objective (descending).
        """
        # Get all occupied indices
        occupied_indices = np.where(self._archive.occupied)[0]

        if len(occupied_indices) == 0:
            return []

        # Get objectives for sorting
        objectives = self._archive.data("objective")[occupied_indices]
        sorted_indices = occupied_indices[np.argsort(objectives)[::-1]]

        if n is not None:
            sorted_indices = sorted_indices[:n]

        elites = []
        for idx in sorted_indices:
            entry = self.get_elite(int(idx))
            if entry is not None:
                elites.append(entry)

        return elites

    def sample_elites(self, n: int, seed: int | None = None) -> list[PromptEntry]:
        """Sample n random elites from the archive.

        Args:
            n: Number of elites to sample.
            seed: Random seed.

        Returns:
            List of sampled PromptEntry objects.
        """
        if seed is not None:
            np.random.seed(seed)

        occupied_indices = np.where(self._archive.occupied)[0]

        if len(occupied_indices) == 0:
            return []

        # Sample indices
        n = min(n, len(occupied_indices))
        sampled_indices = np.random.choice(occupied_indices, size=n, replace=False)

        elites = []
        for idx in sampled_indices:
            entry = self.get_elite(int(idx))
            if entry is not None:
                elites.append(entry)

        return elites

    @property
    def stats(self) -> dict:
        """Get archive statistics."""
        return {
            "num_elites": int(self._archive.stats.num_elites),
            "coverage": float(self._archive.stats.coverage),
            "qd_score": float(self._archive.stats.qd_score),
            "obj_max": float(self._archive.stats.obj_max),
            "obj_mean": float(self._archive.stats.obj_mean),
        }

    def save(self, path: Path | str) -> None:
        """Save the archive to disk.

        Args:
            path: Path to save the archive.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "archive": self._archive,
            "prompt_data": self._prompt_data,
            "next_id": self._next_id,
            "config": self.config,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls,
        path: Path | str,
        embedding_space: EmbeddingSpace,
    ) -> "BehaviorArchive":
        """Load an archive from disk.

        Args:
            path: Path to the saved archive.
            embedding_space: Embedding space instance.

        Returns:
            Loaded BehaviorArchive.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.config = data["config"]
        instance.embedding_space = embedding_space
        instance._archive = data["archive"]
        instance._prompt_data = data["prompt_data"]
        instance._next_id = data["next_id"]

        return instance

    def to_dataframe(self):
        """Export archive data to a pandas DataFrame.

        Returns:
            DataFrame with prompt data and measures.
        """
        import pandas as pd

        rows = []
        for entry in self.get_elites():
            rows.append({
                "prompt": entry.prompt,
                "objective": entry.objective,
                "pca_1": entry.measures.pca_1,
                "pca_2": entry.measures.pca_2,
                "variance": entry.measures.variance,
                "response": entry.response,
                "reasoning": entry.reasoning,
            })

        return pd.DataFrame(rows)
