import os
from typing import List, Optional, Dict, Any
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import logging
import shutil

logger = logging.getLogger(__name__)


DEFAULT_CHROMA_DB_PATH = "chroma_db_store"
DEFAULT_COLLECTION_NAME = "annual_reports_collection"


def create_and_persist_chroma_db(chunks: List[Document], embeddings: Embeddings, db_path: str = DEFAULT_CHROMA_DB_PATH, collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[Chroma]:
    if not chunks:
        logger.warning("No chunks provided to create ChromaDB. Aborting.")
        return None
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=collection_name)
        logger.info(f"ChromaDB vector store created/updated with {len(chunks)} chunks in collection '{collection_name}' at '{db_path}'.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating/persisting ChromaDB vector store: {e}", exc_info=True)
        return None

def load_chroma_db(embeddings: Embeddings, db_path: str = DEFAULT_CHROMA_DB_PATH, collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[Chroma]:
    if not os.path.exists(db_path) or not os.listdir(db_path):
        logger.info(f"ChromaDB persistence directory not found or empty at '{db_path}'. Cannot load collection '{collection_name}'.")
        return None
    try:
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=collection_name)
        if vector_store._collection.count() == 0:
            logger.warning(f"ChromaDB collection '{collection_name}' at '{db_path}' loaded but is empty.")
            return vector_store
        logger.info(f"ChromaDB vector store loaded successfully from collection '{collection_name}' at '{db_path}'. Count: {vector_store._collection.count()}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading ChromaDB vector store from collection '{collection_name}' at '{db_path}': {e}", exc_info=True)
        return None

def delete_chroma_collection(db_path: str = DEFAULT_CHROMA_DB_PATH, collection_name: str = DEFAULT_COLLECTION_NAME):
    if not os.path.exists(db_path):
        logger.info(f"ChromaDB path '{db_path}' not found. Nothing to delete for collection '{collection_name}'.")
        return
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' deleted successfully from '{db_path}'.")
        else:
            logger.info(f"ChromaDB collection '{collection_name}' not found in '{db_path}'. Nothing to delete.")
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collection '{collection_name}' from '{db_path}': {e}", exc_info=True)


def delete_entire_chroma_db_directory(db_path: str = DEFAULT_CHROMA_DB_PATH):
    if os.path.exists(db_path) and os.path.isdir(db_path):
        try:
            shutil.rmtree(db_path)
            logger.info(f"Entire ChromaDB directory at '{db_path}' deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB directory at '{db_path}': {e}", exc_info=True)
    else:
        logger.info(f"ChromaDB directory at '{db_path}' not found, nothing to delete.")