"""Chunk the guideline corpus and embed it into a local Chroma store.

Chunking strategy: recursive character splitting, chunk_size=800 chars /
overlap=100. The corpus is short prose documents with markdown headers
(WHO/AAD/EPA/SkinCancer.org guidance) — 800 chars keeps each chunk to
roughly one sub-topic (a few sentences to a short section), which is
small enough for precise retrieval but large enough to stay coherent
without the header's context. Simple and good enough for a ~20-chunk
corpus; documented as the deliberate choice in ARCHITECTURE.md.

Run: python -m app.rag.ingest
"""
from __future__ import annotations

import glob
import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config


def ingest() -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for path in glob.glob(os.path.join(config.CORPUS_DIR, "*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for chunk in splitter.split_text(text):
            docs.append({"text": chunk, "source": os.path.basename(path)})

    if not docs:
        print("No corpus files found in", config.CORPUS_DIR)
        return 0

    from langchain_core.documents import Document

    documents = [Document(page_content=d["text"], metadata={"source": d["source"]}) for d in docs]

    Chroma.from_documents(
        documents=documents,
        embedding=config.get_embeddings(),
        persist_directory=config.CHROMA_DIR,
        collection_name="sun_guidelines",
    )
    print(f"Ingested {len(documents)} chunks into {config.CHROMA_DIR}")
    return len(documents)


if __name__ == "__main__":
    ingest()
