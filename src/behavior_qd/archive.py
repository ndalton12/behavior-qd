"""Archive management for behavior-qd framework.

Wraps pyribs CVTArchive with behavior-specific functionality.
Includes ContinuationArchive for multi-turn QD using GridArchive.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ribs.archives import AddStatus, CVTArchive, GridArchive
from scipy.cluster.vq import kmeans

from behavior_qd.config import ArchiveConfig, MultiTurnConfig
from behavior_qd.conversation import Conversation
from behavior_qd.embeddings import EmbeddingSpace, Measures


def _safe_archive_stats(archive) -> dict:
    """Extract stats from a pyribs archive, handling empty archives.

    When the archive is empty, ``obj_max`` and ``obj_mean`` are ``None``
    in pyribs 0.9+.  This helper converts them to ``0.0`` so downstream
    code can always treat the values as floats.
    """
    raw = archive.stats
    return {
        "num_elites": int(raw.num_elites),
        "coverage": float(raw.coverage),
        "qd_score": float(raw.qd_score),
        "obj_max": float(raw.obj_max) if raw.obj_max is not None else 0.0,
        "obj_mean": float(raw.obj_mean) if raw.obj_mean is not None else 0.0,
    }


@dataclass
class PromptEntry:
    """An entry in the archive representing an elicited prompt.

    For single-turn (Turn 0): ``prompt`` holds the prompt text,
    ``response`` holds the target model's response.

    For multi-turn (Turn 1+): ``prompt`` holds the latest continuation text,
    ``conversation`` holds the full conversation thread.
    """

    prompt: str
    objective: float
    measures: Measures
    response: str | None = None
    reasoning: str | None = None
    metadata: dict | None = None
    conversation: Conversation | None = None


class BehaviorArchive:
    """Archive for storing diverse behavior-eliciting prompts.

    Uses CVTArchive with 2D measures:
    - dim1: Random projection dimension 1 (sentence embedding)
    - dim2: Random projection dimension 2 (sentence embedding)
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

        # Define measure ranges (2D: dim1, dim2)
        ranges = [
            config.dim_range,  # dim1
            config.dim_range,  # dim2
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
            measures_list = [self.embedding_space.compute_measures(p) for p in prompts]

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

    def get_elite_by_entry_id(self, entry_id: int) -> PromptEntry | None:
        """Get an elite entry by its entry ID.

        Args:
            entry_id: Entry ID in prompt_data.

        Returns:
            PromptEntry or None if not found.
        """
        return self._prompt_data.get(entry_id)

    def get_elites(self, n: int | None = None) -> list[PromptEntry]:
        """Get top n elites by objective score.

        Args:
            n: Number of elites to return. None for all.

        Returns:
            List of PromptEntry sorted by objective (descending).
        """
        # In pyribs 0.9+, iterate directly over archive to get all elites
        # Each elite is a dict with keys: 'index', 'solution', 'objective', 'measures', 'threshold'
        elite_data = []
        for elite in self._archive:
            entry_id = int(elite["solution"][0])
            entry = self._prompt_data.get(entry_id)
            if entry is not None:
                elite_data.append((elite["objective"], entry))

        if not elite_data:
            return []

        # Sort by objective (descending)
        elite_data.sort(key=lambda x: x[0], reverse=True)

        if n is not None:
            elite_data = elite_data[:n]

        return [entry for _, entry in elite_data]

    def sample_elites(self, n: int, seed: int | None = None) -> list[PromptEntry]:
        """Sample n random elites from the archive.

        Args:
            n: Number of elites to sample.
            seed: Random seed.

        Returns:
            List of sampled PromptEntry objects.
        """
        # Get all elites first
        all_elites = self.get_elites()

        if not all_elites:
            return []

        # Sample
        rng = np.random.default_rng(seed)
        n = min(n, len(all_elites))
        indices = rng.choice(len(all_elites), size=n, replace=False)

        return [all_elites[i] for i in indices]

    @property
    def stats(self) -> dict:
        """Get archive statistics."""
        return _safe_archive_stats(self._archive)

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
            rows.append(
                {
                    "prompt": entry.prompt,
                    "objective": entry.objective,
                    "dim1": entry.measures.dim1,
                    "dim2": entry.measures.dim2,
                    "response": entry.response,
                    "reasoning": entry.reasoning,
                }
            )

        return pd.DataFrame(rows)


class ContinuationArchive:
    """Archive for multi-turn continuations using GridArchive.

    One axis is the parent conversation index (discrete), the remaining
    axes are embedding-based diversity measures (dim1, dim2) computed on
    the continuation text. This ensures diverse follow-ups across and
    within each parent conversation.
    """

    def __init__(
        self,
        num_parents: int,
        embedding_space: EmbeddingSpace,
        archive_config: ArchiveConfig,
        multi_turn_config: MultiTurnConfig,
    ):
        """Initialize the continuation archive.

        Args:
            num_parents: Number of parent conversations from previous turn.
            embedding_space: Embedding space for computing continuation measures.
            archive_config: Base archive configuration (for measure ranges).
            multi_turn_config: Multi-turn specific configuration.
        """
        self.num_parents = num_parents
        self.embedding_space = embedding_space
        self.archive_config = archive_config
        self.multi_turn_config = multi_turn_config

        bins = multi_turn_config.grid_bins_per_dim

        # Grid dimensions: [parent_id, dim1, dim2]
        self._archive = GridArchive(
            solution_dim=1,  # Store entry ID
            dims=[num_parents, bins, bins],
            ranges=[
                (0, num_parents),  # Parent index
                archive_config.dim_range,  # dim1
                archive_config.dim_range,  # dim2
            ],
        )

        self._prompt_data: dict[int, PromptEntry] = {}
        self._next_id = 0

        # Map from parent index → parent PromptEntry (set during setup)
        self._parent_entries: dict[int, PromptEntry] = {}

    @property
    def archive(self) -> GridArchive:
        """Access the underlying pyribs GridArchive."""
        return self._archive

    def set_parent_entries(self, parents: dict[int, PromptEntry]) -> None:
        """Store parent conversation entries for reference.

        Args:
            parents: Mapping from parent index to PromptEntry.
        """
        self._parent_entries = parents

    def get_parent_entry(self, parent_idx: int) -> PromptEntry | None:
        """Get the parent conversation entry for a given index.

        Args:
            parent_idx: Index into the parent conversations.

        Returns:
            PromptEntry or None.
        """
        return self._parent_entries.get(parent_idx)

    def add(
        self,
        prompt: str,
        objective: float,
        parent_idx: int,
        measures: Measures | None = None,
        response: str | None = None,
        reasoning: str | None = None,
        conversation: Conversation | None = None,
        metadata: dict | None = None,
    ) -> tuple[int, int | None]:
        """Add a continuation to the archive.

        Args:
            prompt: The continuation text (follow-up user message).
            objective: Objective score.
            parent_idx: Index of the parent conversation.
            measures: Pre-computed measures on the continuation, or None to compute.
            response: Target model response to the full conversation.
            reasoning: Judge reasoning.
            conversation: Full conversation thread.
            metadata: Additional metadata.

        Returns:
            Tuple of (status, cell_index or None).
        """
        if measures is None:
            measures = self.embedding_space.compute_measures(prompt)

        entry = PromptEntry(
            prompt=prompt,
            objective=objective,
            measures=measures,
            response=response,
            reasoning=reasoning,
            conversation=conversation,
            metadata=metadata,
        )

        entry_id = self._next_id
        self._next_id += 1

        # Measures for GridArchive: [parent_idx, dim1, dim2]
        grid_measures = np.array(
            [parent_idx + 0.5, measures.dim1, measures.dim2],
            dtype=np.float32,
        )

        result = self._archive.add_single(
            solution=np.array([entry_id]),
            objective=objective,
            measures=grid_measures,
        )

        status = result["status"] if isinstance(result, dict) else result
        was_added = int(status) > 0

        if was_added:
            self._prompt_data[entry_id] = entry

        return int(status), entry_id if was_added else None

    def add_batch(
        self,
        prompts: list[str],
        objectives: list[float],
        parent_indices: list[int],
        measures_list: list[Measures] | None = None,
        responses: list[str | None] | None = None,
        reasonings: list[str | None] | None = None,
        conversations: list[Conversation | None] | None = None,
    ) -> list[int]:
        """Add a batch of continuations to the archive.

        Args:
            prompts: Continuation texts.
            objectives: Objective scores.
            parent_indices: Parent conversation indices.
            measures_list: Pre-computed measures on continuations.
            responses: Target model responses.
            reasonings: Judge reasonings.
            conversations: Full conversation threads.

        Returns:
            List of status codes for each entry.
        """
        n = len(prompts)

        if measures_list is None:
            measures_list = [self.embedding_space.compute_measures(p) for p in prompts]
        if responses is None:
            responses = [None] * n
        if reasonings is None:
            reasonings = [None] * n
        if conversations is None:
            conversations = [None] * n

        solutions = []
        objs = []
        meas = []
        entry_ids = []
        entries = []

        for i in range(n):
            entry = PromptEntry(
                prompt=prompts[i],
                objective=objectives[i],
                measures=measures_list[i],
                response=responses[i],
                reasoning=reasonings[i],
                conversation=conversations[i],
            )
            entry_id = self._next_id + i
            entry_ids.append(entry_id)
            entries.append(entry)

            solutions.append([entry_id])
            objs.append(objectives[i])
            # Grid measures: [parent_idx (centered in bin), dim1, dim2]
            meas.append(
                [
                    parent_indices[i] + 0.5,
                    measures_list[i].dim1,
                    measures_list[i].dim2,
                ]
            )

        self._next_id += n

        result = self._archive.add(
            solution=np.array(solutions),
            objective=np.array(objs),
            measures=np.array(meas, dtype=np.float32),
        )

        status_array = result["status"] if isinstance(result, dict) else result

        for entry_id, entry, status in zip(entry_ids, entries, status_array):
            if int(status) > 0:
                self._prompt_data[entry_id] = entry

        return [int(s) for s in status_array]

    def get_elites(self, n: int | None = None) -> list[PromptEntry]:
        """Get top n elites by objective score.

        Args:
            n: Number of elites to return. None for all.

        Returns:
            List of PromptEntry sorted by objective (descending).
        """
        elite_data = []
        for elite in self._archive:
            entry_id = int(elite["solution"][0])
            entry = self._prompt_data.get(entry_id)
            if entry is not None:
                elite_data.append((elite["objective"], entry))

        if not elite_data:
            return []

        elite_data.sort(key=lambda x: x[0], reverse=True)

        if n is not None:
            elite_data = elite_data[:n]

        return [entry for _, entry in elite_data]

    def sample_elites(self, n: int, seed: int | None = None) -> list[PromptEntry]:
        """Sample n random elites from the archive.

        Args:
            n: Number of elites to sample.
            seed: Random seed.

        Returns:
            List of sampled PromptEntry objects.
        """
        all_elites = self.get_elites()
        if not all_elites:
            return []

        rng = np.random.default_rng(seed)
        n = min(n, len(all_elites))
        indices = rng.choice(len(all_elites), size=n, replace=False)
        return [all_elites[i] for i in indices]

    def get_elites_for_parent(self, parent_idx: int) -> list[PromptEntry]:
        """Get all elites that continue a specific parent conversation.

        Args:
            parent_idx: Parent conversation index.

        Returns:
            List of PromptEntry for that parent, sorted by objective.
        """
        elite_data = []
        for elite in self._archive:
            # Check if this elite belongs to the given parent
            measures = elite["measures"]
            elite_parent = int(measures[0])
            if elite_parent == parent_idx:
                entry_id = int(elite["solution"][0])
                entry = self._prompt_data.get(entry_id)
                if entry is not None:
                    elite_data.append((elite["objective"], entry))

        elite_data.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in elite_data]

    @property
    def stats(self) -> dict:
        """Get archive statistics."""
        s = _safe_archive_stats(self._archive)
        s["num_parents"] = self.num_parents
        return s

    def save(self, path: Path | str) -> None:
        """Save the continuation archive to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "archive": self._archive,
            "prompt_data": self._prompt_data,
            "next_id": self._next_id,
            "parent_entries": self._parent_entries,
            "num_parents": self.num_parents,
            "archive_config": self.archive_config,
            "multi_turn_config": self.multi_turn_config,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls,
        path: Path | str,
        embedding_space: EmbeddingSpace,
    ) -> ContinuationArchive:
        """Load a continuation archive from disk.

        Args:
            path: Path to the saved archive.
            embedding_space: Embedding space instance.

        Returns:
            Loaded ContinuationArchive.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.num_parents = data["num_parents"]
        instance.embedding_space = embedding_space
        instance.archive_config = data["archive_config"]
        instance.multi_turn_config = data["multi_turn_config"]
        instance._archive = data["archive"]
        instance._prompt_data = data["prompt_data"]
        instance._next_id = data["next_id"]
        instance._parent_entries = data["parent_entries"]

        return instance

    def to_dataframe(self):
        """Export archive data to a pandas DataFrame."""
        import pandas as pd

        rows = []
        for entry in self.get_elites():
            row = {
                "prompt": entry.prompt,
                "objective": entry.objective,
                "dim1": entry.measures.dim1,
                "dim2": entry.measures.dim2,
                "response": entry.response,
                "reasoning": entry.reasoning,
            }
            if entry.conversation:
                row["num_turns"] = entry.conversation.num_turns
                row["parent_id"] = entry.conversation.parent_id
                row["full_conversation"] = entry.conversation.format_for_display(200)
            rows.append(row)

        return pd.DataFrame(rows)
