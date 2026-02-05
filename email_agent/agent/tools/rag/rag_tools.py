"""RAG tool for searching operational guidance."""

import hashlib
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .bge_embeddings import BGEEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en"
COLLECTION_NAME = "guidance"
VECTOR_SIZE = 384  # BGE-small-en embedding dimension

# Paths
_module_dir = Path(__file__).parent
_vector_db_path = _module_dir / "guidance_db"
_docs_folder = _module_dir / "guidance_docs"

# Initialize client and embeddings
_qdrant_client = QdrantClient(path=str(_vector_db_path))
_embeddings = BGEEmbeddings(model_name=EMBEDDING_MODEL)


def _ensure_collection() -> None:
    """Create the collection if it doesn't exist."""
    try:
        _qdrant_client.get_collection(COLLECTION_NAME)
    except ValueError:
        # Collection doesn't exist, create it
        _qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created guidance collection")


def _get_indexed_hashes() -> set[str]:
    """Get hashes of already-indexed documents."""
    try:
        results, _ = _qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
        )
        return {p.payload.get("metadata", {}).get("hash", "") for p in results}
    except Exception:
        return set()


def _load_docs_from_folder(store: QdrantVectorStore) -> None:
    """Load documents from guidance_docs folder if not already indexed."""
    if not _docs_folder.exists():
        _docs_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Created guidance_docs folder at %s", _docs_folder)
        return

    # Get already indexed hashes
    indexed = _get_indexed_hashes()

    # Find files to index
    docs_to_add = []
    for file_path in _docs_folder.glob("*"):
        if file_path.suffix.lower() not in (".txt", ".md"):
            continue

        content = file_path.read_text(encoding="utf-8")
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in indexed:
            continue

        docs_to_add.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name, "hash": content_hash},
            )
        )

    if docs_to_add:
        store.add_documents(docs_to_add)
        logger.info("Indexed %d new guidance documents", len(docs_to_add))


# Initialize: ensure collection exists, create store, load docs
_ensure_collection()
_vector_store = QdrantVectorStore(
    client=_qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=_embeddings,
)
_load_docs_from_folder(_vector_store)


class SearchGuidanceInput(BaseModel):
    """Input schema for guidance search tool."""

    query: str = Field(description="Query string to search guidance database")
    max_results: int = Field(default=2, description="Maximum number of results to return")


@tool(args_schema=SearchGuidanceInput)
def search_guidance_tool(query: str, max_results: int = 2) -> str:
    """Search operational knowledge base for relevant guidance and documentation."""
    display_query = query[:77] + "..." if len(query) > 80 else query
    logger.info("Searching guidance for: %s (max_results=%d)", display_query, max_results)
    results = _vector_store.similarity_search(query, k=max_results)
    logger.debug(f"Found {len(results)} results for query: {query}")
    return "\n\n".join([f"**{doc.metadata.get('source', 'Unknown')}**\n{doc.page_content}" for doc in results])
