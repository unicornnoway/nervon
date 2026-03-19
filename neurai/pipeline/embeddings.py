from __future__ import annotations

import logging

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class _LiteLLMStub:
        @staticmethod
        def embedding(*args, **kwargs):
            raise ModuleNotFoundError("litellm is required for embedding requests")

    litellm = _LiteLLMStub()

logger = logging.getLogger(__name__)


def get_embedding(text: str, model: str) -> list[float]:
    try:
        response = litellm.embedding(model=model, input=[text])
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Embedding request failed: %s", exc)
        return []

    data = getattr(response, "data", None) or []
    if not data:
        logger.warning("Embedding response contained no vectors")
        return []

    embedding = getattr(data[0], "embedding", None)
    if not embedding:
        logger.warning("Embedding response was missing vector data")
        return []

    return [float(value) for value in embedding]


def get_embeddings(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []

    try:
        response = litellm.embedding(model=model, input=texts)
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        logger.warning("Batch embedding request failed: %s", exc)
        return []

    data = getattr(response, "data", None) or []
    if not data:
        logger.warning("Batch embedding response contained no vectors")
        return []

    embeddings: list[list[float]] = []
    for item in data:
        vector = getattr(item, "embedding", None)
        if not vector:
            logger.warning("Batch embedding response had a missing vector")
            return []
        embeddings.append([float(value) for value in vector])

    return embeddings
