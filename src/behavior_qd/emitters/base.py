"""Base emitter interface for behavior-qd framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from behavior_qd.archive import BehaviorArchive
from behavior_qd.embeddings import Measures


@dataclass
class EmitterResult:
    """Result from an emitter's ask() call."""

    prompts: list[str]
    # For embedding emitter, store the genomes for tell()
    genomes: NDArray[np.float32] | None = None


@dataclass
class EmitterFeedback:
    """Feedback to provide to an emitter's tell() call."""

    prompts: list[str]
    objectives: list[float]
    measures: list[Measures]
    # Status from archive add
    statuses: list[int] | None = None


class BaseEmitter(ABC):
    """Abstract base class for emitters.

    Emitters generate candidate prompts for evaluation.
    """

    def __init__(self, archive: BehaviorArchive):
        """Initialize the emitter.

        Args:
            archive: The behavior archive instance.
        """
        self.archive = archive

    @abstractmethod
    def ask(self) -> EmitterResult:
        """Generate a batch of candidate prompts.

        Returns:
            EmitterResult containing prompts and optional metadata.
        """
        pass

    @abstractmethod
    def tell(self, feedback: EmitterFeedback) -> None:
        """Update emitter state based on evaluation feedback.

        Args:
            feedback: Feedback from evaluating the generated prompts.
        """
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of prompts generated per ask() call."""
        pass
