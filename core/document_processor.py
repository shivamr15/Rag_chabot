import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
import tempfile
import logging

logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".pdf": {"loader": PyPDFLoader, "kwargs": {}},
    ".docx": {"loader": UnstructuredWordDocumentLoader, "kwargs": {}},
    ".pptx": {"loader": UnstructuredPowerPointLoader, "kwargs": {}},
    ".txt": {"loader": TextLoader, "kwargs": {"encoding": "utf8"}},
    ".png": {"loader": UnstructuredImageLoader, "kwargs": {"mode": "single"}},
    ".jpg": {"loader": UnstructuredImageLoader, "kwargs": {"mode": "single"}},
    ".jpeg": {"loader": UnstructuredImageLoader, "kwargs": {"mode": "single"}},
}

def load_single_document(file_path: str) -> List[Document]:
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_info = LOADER_MAPPING.get(file_extension)
    if not loader_info:
        logger.warning(f"Unsupported file type: {file_extension}. Skipping {file_path}")
        return []
    loader_class = loader_info["loader"]
    loader_kwargs = loader_info["kwargs"]
    try:
        loader = loader_class(file_path, **loader_kwargs)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading {file_path} with {loader_class.__name__}: {e}")
        if file_extension == ".pdf":
            logger.info(f"Trying UnstructuredPDFLoader with OCR as fallback for PDF: {file_path}")
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(file_path, mode="single", strategy="ocr_only")
                return loader.load()
            except Exception as e_unstructured:
                logger.error(f"UnstructuredPDFLoader fallback also failed for {file_path}: {e_unstructured}")
                return []
        return []


def load_documents_from_uploaded_files(uploaded_files_with_metadata: List[Dict[str, Any]]) -> List[Document]:
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    for item in uploaded_files_with_metadata:
        uploaded_file = item['file']
        company_name_meta = item.get('company_name')
        year_meta = item.get('year')
        file_path = None 
        try:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logger.info(f"Processing file: {uploaded_file.name} with metadata: Company='{company_name_meta}', Year='{year_meta}'")
            docs = load_single_document(file_path)
            if docs:
                for doc in docs:
                    if not isinstance(doc.metadata, dict):
                        doc.metadata = {} 
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["company_name"] = company_name_meta if company_name_meta else "Not Specified"
                    doc.metadata["year"] = str(year_meta) if year_meta else "Not Specified"
                all_docs.extend(docs)
            else:
                logger.warning(f"No documents extracted from {uploaded_file.name}")
        except Exception as e:
            logger.error(f"Failed to process {uploaded_file.name}: {e}", exc_info=True)
        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
    try:
        os.rmdir(temp_dir)
    except OSError as e:
        logger.error(f"Error removing temporary directory {temp_dir}: {e}")
    return all_docs

def chunk_documents(documents: List[Document], embeddings: Embeddings) -> List[Document]:
    if not documents:
        return []
    logger.info(f"Starting semantic chunking for {len(documents)} document(s)...")
    text_splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="percentile",
    )
    all_chunks = []
    for i, doc in enumerate(documents):
        logger.info(f"Semantic chunking document {i+1}/{len(documents)}: {doc.metadata.get('source', 'Unknown source')}")
        try:
            semantic_chunks_text = text_splitter.split_text(doc.page_content)
            current_doc_chunks = []
            for chunk_text in semantic_chunks_text:
                new_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                current_doc_chunks.append(new_doc)
            all_chunks.extend(current_doc_chunks)
            logger.info(f"Document {doc.metadata.get('source', 'N/A')} split into {len(current_doc_chunks)} semantic chunks.")
        except Exception as e:
            logger.error(f"Error during semantic chunking for doc {doc.metadata.get('source', 'N/A')}: {e}", exc_info=True)
            if not semantic_chunks_text and len(doc.page_content) < 1000:
                 all_chunks.append(doc)
                 logger.warning(f"Doc {doc.metadata.get('source', 'N/A')} was small and not split; added as single chunk.")
    logger.info(f"Finished semantic chunking. Total docs: {len(documents)}, Total chunks: {len(all_chunks)}.")
    return all_chunks