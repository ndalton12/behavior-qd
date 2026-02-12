"""Embedding system for behavior-qd framework.

Handles sentence embeddings, token embeddings, random projection measures,
PCA projection on vocab matrix, and temperature-based top-k token snapping.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from behavior_qd.config import EmbeddingConfig

# Fixed seed for the random projection matrix so measures are deterministic
_RANDOM_PROJECTION_SEED = 42


@dataclass
class Measures:
    """Archive measures for a prompt."""

    pca_1: float
    pca_2: float
    variance: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return measures as a tuple for pyribs."""
        return (self.pca_1, self.pca_2, self.variance)

    def as_array(self) -> NDArray[np.float32]:
        """Return measures as numpy array."""
        return np.array([self.pca_1, self.pca_2, self.variance], dtype=np.float32)


class EmbeddingSpace:
    """Manages embeddings, random projection measures, PCA, and token snapping.

    Archive measures use random projections on sentence embeddings (corpus-free).
    PCA on the vocabulary matrix is retained for the embedding emitter's token snapping.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize the embedding space.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._model: SentenceTransformer | None = None
        self._pca: PCA | None = None
        self._vocab_embeddings: NDArray[np.float32] | None = None

    @cached_property
    def model(self) -> SentenceTransformer:
        """Lazily load the sentence transformer model."""
        return SentenceTransformer(self.config.model_name, device=self.config.device)

    @cached_property
    def tokenizer(self):
        """Get the tokenizer from the model."""
        # sentence-transformers wraps a transformer model
        return self.model.tokenizer

    @cached_property
    def vocab_embeddings(self) -> NDArray[np.float32]:
        """Get the vocabulary embedding matrix from the model.

        Returns:
            Array of shape (vocab_size, embed_dim)
        """
        # Access the underlying transformer model's embeddings
        transformer = self.model[0].auto_model
        embed_layer = transformer.embeddings.word_embeddings
        weights = embed_layer.weight.detach().cpu().numpy()
        return weights.astype(np.float32)

    @cached_property
    def pca(self) -> PCA:
        """Fit PCA on the vocabulary embedding matrix.

        This is deterministic for a given model - no corpus fitting needed.
        """
        pca = PCA(n_components=self.config.pca_components)
        pca.fit(self.vocab_embeddings)
        return pca

    @cached_property
    def random_projection_matrix(self) -> NDArray[np.float32]:
        """Fixed Gaussian random projection matrix for sentence embedding → 2D.

        Uses a Gaussian random matrix scaled by 1/sqrt(2) per
        Johnson-Lindenstrauss. Deterministic for a given seed.
        Prioritizes spread over strict distance preservation.
        """
        rng = np.random.default_rng(_RANDOM_PROJECTION_SEED)
        sentence_dim = self.model.get_sentence_embedding_dimension()
        matrix = rng.standard_normal((sentence_dim, 2)).astype(np.float32)
        matrix /= np.sqrt(2)
        return matrix

    @cached_property
    def embed_dim(self) -> int:
        """Dimensionality of the embedding space."""
        return self.vocab_embeddings.shape[1]

    @cached_property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return self.vocab_embeddings.shape[0]

    def encode_sentence(self, text: str) -> NDArray[np.float32]:
        """Encode a sentence to a single embedding vector.

        Args:
            text: Input text to encode.

        Returns:
            Embedding vector of shape (embed_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def encode_tokens(self, text: str) -> NDArray[np.float32]:
        """Encode text and return token-level embeddings.

        Uses the model's word embeddings for each token in the input.

        Args:
            text: Input text to encode.

        Returns:
            Token embeddings of shape (num_tokens, embed_dim)
        """
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].squeeze(0)

        # Get embeddings for each token from the vocab matrix
        token_embeds = self.vocab_embeddings[input_ids.numpy()]
        return token_embeds

    def compute_measures(self, text: str) -> Measures:
        """Compute archive measures for a prompt.

        Measures:
        - pca_1, pca_2: Random projection of sentence embedding to 2D,
          normalized to [-1, 1] via tanh
        - variance: "Distance traveled" — sum of adjacent Euclidean distances
          between consecutive token embeddings, normalized via tanh

        Uses sentence-level embedding with a fixed random projection matrix
        (no corpus fitting needed). The random projection preserves pairwise
        distances between prompts (Johnson-Lindenstrauss).

        Args:
            text: Input prompt text.

        Returns:
            Measures object with pca_1, pca_2, and variance.
        """
        # Sentence embedding → random projection for 2D measures
        sentence_embed = self.encode_sentence(text)  # (sentence_dim,)
        projected = sentence_embed @ self.random_projection_matrix  # (2,)

        # Normalize to [-1, 1] via tanh so values always fit archive range
        proj_normalized = np.tanh(projected)

        # "Distance traveled" — sum of Euclidean distances between
        # consecutive token embeddings. Captures how much the prompt
        # jumps around in embedding space from token to token.
        token_embeds = self.encode_tokens(text)
        if len(token_embeds) > 1:
            diffs = np.diff(token_embeds, axis=0)  # (num_tokens-1, embed_dim)
            distances = np.linalg.norm(diffs, axis=1)  # (num_tokens-1,)
            distance_traveled = float(distances.sum())
        else:
            distance_traveled = 0.0

        # Normalize distance traveled to [0, 1] via scaled tanh
        # Scale factor chosen so typical prompts spread across [0, 1]
        distance_normalized = float(np.tanh(distance_traveled / 50.0))

        return Measures(
            pca_1=float(proj_normalized[0]),
            pca_2=float(proj_normalized[1]),
            variance=distance_normalized,
        )

    def snap_to_tokens(
        self,
        embedding_sequence: NDArray[np.float32],
        top_k: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Snap continuous embeddings to discrete tokens.

        Uses temperature-based top-k sampling for exploration.

        Args:
            embedding_sequence: Embeddings of shape (seq_len, embed_dim)
            top_k: Number of nearest tokens to consider. Defaults to config.
            temperature: Sampling temperature. 0 = deterministic argmax.
            seed: Random seed for reproducibility.

        Returns:
            Decoded text from snapped tokens.
        """
        top_k = top_k if top_k is not None else self.config.top_k
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )

        if seed is not None:
            np.random.seed(seed)

        token_ids = []

        for pos_embedding in embedding_sequence:
            # Compute cosine similarities to all vocab tokens
            # Normalize for cosine similarity
            pos_norm = pos_embedding / (np.linalg.norm(pos_embedding) + 1e-8)
            vocab_norms = self.vocab_embeddings / (
                np.linalg.norm(self.vocab_embeddings, axis=1, keepdims=True) + 1e-8
            )
            similarities = vocab_norms @ pos_norm

            # Get top-k most similar tokens
            top_k_indices = np.argsort(similarities)[-top_k:]
            top_k_sims = similarities[top_k_indices]

            if temperature == 0:
                # Hard snap to nearest (deterministic)
                token_ids.append(int(top_k_indices[-1]))
            else:
                # Temperature-scaled softmax sampling
                logits = top_k_sims / temperature
                probs = softmax(logits)
                selected = np.random.choice(top_k_indices, p=probs)
                token_ids.append(int(selected))

        # Decode token IDs to text
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def random_embedding_sequence(
        self,
        length: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate a random embedding sequence for initialization.

        Samples random points within the convex hull of vocab embeddings.

        Args:
            length: Number of token positions.
            seed: Random seed for reproducibility.

        Returns:
            Random embeddings of shape (length, embed_dim)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample random vocab embeddings and interpolate
        # This keeps us within the "reasonable" embedding space
        indices = np.random.randint(0, self.vocab_size, size=(length, 2))
        weights = np.random.random(size=(length, 1))

        embeddings = (
            weights * self.vocab_embeddings[indices[:, 0]]
            + (1 - weights) * self.vocab_embeddings[indices[:, 1]]
        )

        return embeddings.astype(np.float32)

    def embedding_sequence_with_length(
        self,
        genome: NDArray[np.float32],
        max_length: int | None = None,
    ) -> tuple[NDArray[np.float32], int]:
        """Extract embedding sequence and length from a genome.

        The genome has structure: [embed_0, embed_1, ..., embed_{max_len-1}, length_param]
        where length_param ∈ [0, 1] maps to actual length.

        Args:
            genome: Flattened genome array.
            max_length: Maximum sequence length. Defaults to config.

        Returns:
            Tuple of (active embeddings, actual length)
        """
        max_length = max_length or self.config.max_prompt_length

        # Last element is length parameter
        length_param = genome[-1]
        # Clamp to [0, 1]
        length_param = np.clip(length_param, 0.0, 1.0)

        # Map to actual length (minimum 3 tokens)
        actual_length = max(3, round(length_param * max_length))

        # Reshape embeddings
        embeddings = genome[:-1].reshape(max_length, self.embed_dim)

        # Return only active positions
        return embeddings[:actual_length], actual_length

    def genome_size(self, max_length: int | None = None) -> int:
        """Calculate the genome size for CMA-ME.

        Args:
            max_length: Maximum sequence length. Defaults to config.

        Returns:
            Total genome size (embeddings + length param).
        """
        max_length = max_length or self.config.max_prompt_length
        return max_length * self.embed_dim + 1  # +1 for length parameter
