"""Thin wrapper around FlagEmbedding's BGE-small-en model.

Exposes a LangChain compatible Embeddings implementation for RAG indexing and retrieval.
"""

import logging
from typing import List

from FlagEmbedding import FlagModel
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEEmbeddings(Embeddings):
    """Local BGE-small-en embeddings using FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        use_fp16: bool = False,
        batch_size: int = 32,
    ) -> None:
        """Initialise the BGE embeddings model.

        Args:
            model_name: Hugging Face model name to load.
            use_fp16: Whether to use 16‑bit floats.
            batch_size: Number of texts to encode per batch.
        """
        # On CPU/M2, use_fp16=False is safe; you can experiment with True later.
        logger.info("Initialising BGE embeddings model: %s", model_name)
        self._model = FlagModel(model_name, use_fp16=use_fp16)
        logger.info("BGE model initialised successfully.")
        self._batch_size = batch_size

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        """Encode a batch of texts as query or passage embeddings.

        Args:
            texts: List of raw text strings to embed.
            is_query: If True, use the query prefix; otherwise use passage prefix.

        Returns:
            A list of embedding vectors, one per input text.
        """
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        return self._model.encode(
            prefixed,
            batch_size=self._batch_size,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: The raw text string to embed.

        Returns:
            A list representing the embedding vector for the query.
        """
        return self._embed([text], is_query=True)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed one or more documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            A list of embedding vectors. Empty list if no texts are given.
        """
        if not texts:
            return []
        return self._embed(texts, is_query=False)
