from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from .. import config


@lru_cache
def _store() -> Chroma:
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=config.get_embeddings(),
        collection_name="sun_guidelines",
    )


def retrieve(query: str, k: int = 4):
    """Return top-k relevant chunks (langchain Document objects) for a query."""
    return _store().similarity_search(query, k=k)
