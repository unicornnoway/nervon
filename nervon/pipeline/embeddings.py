from __future__ import annotations

import logging
import os
import time

try:
    import requests as _requests
except ModuleNotFoundError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class _LiteLLMStub:
        @staticmethod
        def embedding(*args, **kwargs):
            raise ModuleNotFoundError("litellm is required for embedding requests")

    litellm = _LiteLLMStub()

logger = logging.getLogger(__name__)

# Default Gemini Embedding config
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 3072


def _is_gemini_model(model: str) -> bool:
    """Check if model string refers to a Gemini embedding model."""
    return "gemini" in model.lower() and "embedding" in model.lower()


def _gemini_model_id(model: str) -> str:
    """Extract the model ID for Gemini API (strip provider prefix)."""
    if model.startswith("gemini/"):
        return model[len("gemini/"):]
    return model


def _get_gemini_api_key() -> str | None:
    """Resolve the Gemini/Google API key from environment."""
    return (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or None
    )


def _embed_gemini(
    text: str,
    model: str,
    task_type: str | None = None,
) -> list[float]:
    """Call Gemini embedding API directly with task_type support."""
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.warning("No GOOGLE_API_KEY or GEMINI_API_KEY found for Gemini embedding")
        return []

    if _requests is None:
        logger.warning("requests library required for Gemini embedding")
        return []

    model_id = _gemini_model_id(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:embedContent?key={api_key}"

    body: dict = {"content": {"parts": [{"text": text}]}}
    if task_type:
        body["taskType"] = task_type

    for attempt in range(5):
        try:
            resp = _requests.post(url, json=body, timeout=30)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                logger.info("Gemini embedding rate limited, waiting %ds (attempt %d/5)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
            return [float(v) for v in values]
        except Exception as exc:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            logger.warning("Gemini embedding request failed: %s", exc)
            return []
    return []


def _batch_embed_gemini(
    texts: list[str],
    model: str,
    task_type: str | None = None,
) -> list[list[float]]:
    """Call Gemini batch embedding API with task_type support."""
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.warning("No GOOGLE_API_KEY or GEMINI_API_KEY found for Gemini embedding")
        return []

    if _requests is None:
        logger.warning("requests library required for Gemini embedding")
        return []

    model_id = _gemini_model_id(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:batchEmbedContents?key={api_key}"

    requests_list = []
    for text in texts:
        req: dict = {
            "model": f"models/{model_id}",
            "content": {"parts": [{"text": text}]},
        }
        if task_type:
            req["taskType"] = task_type
        requests_list.append(req)

    for attempt in range(5):
        try:
            resp = _requests.post(url, json={"requests": requests_list}, timeout=60)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                logger.info("Gemini batch embedding rate limited, waiting %ds (attempt %d/5)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            embeddings = []
            for item in resp.json()["embeddings"]:
                embeddings.append([float(v) for v in item["values"]])
            return embeddings
        except Exception as exc:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            logger.warning("Gemini batch embedding request failed: %s", exc)
            return []
    return []


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str | None = None,
) -> list[float]:
    """Get embedding for a single text. Uses Gemini API directly for Gemini
    models (to support task_type), falls back to litellm for others."""
    if _is_gemini_model(model):
        return _embed_gemini(text, model, task_type)

    # Fallback to litellm for non-Gemini models
    try:
        response = litellm.embedding(model=model, input=[text])
    except Exception as exc:  # pragma: no cover
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


def get_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str | None = None,
) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    if not texts:
        return []

    if _is_gemini_model(model):
        return _batch_embed_gemini(texts, model, task_type)

    # Fallback to litellm for non-Gemini models
    try:
        response = litellm.embedding(model=model, input=texts)
    except Exception as exc:  # pragma: no cover
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
