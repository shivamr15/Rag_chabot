# High-Level Design: RAG Chatbot

## 1. Objective
To build a Retrieval-Augmented Generation (RAG) chatbot using Python, Langchain, Streamlit, and Azure OpenAI (GPT-4o and text-embedding-3-small). The chatbot answers user questions based on the content of multiple company annual reports, supporting PDF, Word (DOCX), PowerPoint (PPTX), TXT, and image formats (via OCR). The system allows users to input optional metadata (company name, year) for each document and supports filtering search results based on this metadata.

## 2. System Goals and Architecture
*   **Goal 1: Accurate Information Retrieval:** Retrieve the most relevant information using semantic search and metadata filtering.
*   **Goal 2: Grounded Generation:** Ensure LLM (GPT-4o) responses are based strictly on the retrieved context.
*   **Goal 3: Multi-Format Support & OCR:** Ingest and process documents in PDF, DOCX, PPTX, TXT, and image formats (PNG, JPG) with OCR capabilities.
*   **Goal 4: User-Friendly Interface:** Provide a Streamlit web interface for document upload, optional metadata input, query submission, and viewing filtered results.
*   **Goal 5: Persistent Data & Filtering:** Store processed data in ChromaDB, allow loading of existing data, and enable users to filter queries by company name and year.
*   **Goal 6: Semantic Understanding:** Employ semantic chunking to improve the relevance of text segments for embedding and retrieval.
*   **Goal 7: Maintainable & Scalable Code:** Modular design for easier maintenance and future enhancements.

### Architectural Diagram

## 3. Component Breakdown

1.  **User Interface (UI - Streamlit):**
    *   Allows users to upload multiple annual reports (PDF, DOCX, PPTX, TXT, PNG, JPG).
    *   Provides input fields for optional metadata (Company Name, Year) for each uploaded file.
    *   Provides an input field for user questions.
    *   Displays chatbot's answers and source document snippets.
    *   Offers dropdowns to filter queries by Company Name and Year based on available metadata in the database.
    *   Manages data source options: use existing persisted data or upload new data (replacing the current collection).

2.  **Document Ingestion & Processing (Langchain):**
    *   **File Loaders:** Uses Langchain document loaders (`PyPDFLoader`, `UnstructuredWordDocumentLoader`, `UnstructuredPowerPointLoader`, `TextLoader`, `UnstructuredImageLoader` for OCR).
    *   **Text Extraction:** Extracts raw text content. `UnstructuredImageLoader` uses Tesseract for OCR.
    *   **Metadata Association:** Associates user-provided Company Name and Year with each document. Defaults to "Not Specified" if left blank by the user.

3.  **Semantic Chunking (Langchain Experimental):**
    *   Uses `SemanticChunker` with the chosen embedding model (Azure OpenAI `text-embedding-3-small`).
    *   Splits extracted text into semantically coherent chunks rather than fixed-size chunks, aiming for better contextual relevance.

4.  **Embedding Generation (Langchain + Azure OpenAI):**
    *   Uses Azure OpenAI's `text-embedding-3-small` via Langchain's `AzureOpenAIEmbeddings`.
    *   Converts each semantic text chunk into a dense vector representation.

5.  **Vector Database (ChromaDB - Langchain Integration):**
    *   **Storage:** Persistently stores generated embeddings along with their corresponding text chunks and associated metadata (source filename, company name, year).
    *   **Collection:** Uses a named collection within ChromaDB (e.g., `annual_reports_collection`).
    *   **Indexing:** ChromaDB automatically indexes embeddings for efficient similarity search.

6.  **Context Retrieval (Langchain + ChromaDB):**
    *   Takes the user's query and selected metadata filters (company, year).
    *   Generates an embedding for the query using the same embedding model.
    *   Performs a similarity search in ChromaDB, applying the specified metadata filters to narrow down the search space to relevant documents.
    *   Retrieves the top-K most relevant text chunks.

7.  **LLM Response Generation (Langchain + Azure OpenAI GPT-4o):**
    *   **Prompt Engineering:** Constructs a prompt for GPT-4o, including the retrieved context (filtered by metadata and similarity) and the user's query.
    *   **LLM Call:** Sends the prompt to Azure OpenAI GPT-4o via `AzureChatOpenAI`.
    *   **Answer Generation:** The LLM generates an answer grounded in the provided context.

## 4. Data Flow and Sequence

**A. Indexing/Ingestion Phase (triggered by "Process Uploaded Documents"):**
1.  User uploads document(s) and provides optional metadata (Company, Year) for each via Streamlit UI.
2.  UI sends documents and their associated metadata to the Ingestion module.
3.  (If replacing data) Existing ChromaDB collection is deleted.
4.  Ingestion module uses appropriate Langchain loaders based on file type (including OCR). User-provided metadata is attached.
5.  Extracted text from each document is passed to the Semantic Chunking module, along with the embedding model.
6.  Semantic chunks are generated.
7.  Chunks are sent to the Embedding Generation module (Azure OpenAI `text-embedding-3-small`).
8.  Generated embeddings, text chunks, and all metadata are stored in the ChromaDB collection.
9.  UI is updated, available metadata filters are refreshed, and RAG chain is initialized/re-initialized.

**B. Querying Phase (triggered by user question):**
1.  User selects optional metadata filters (Company, Year) and submits a question via Streamlit UI.
2.  UI sends the query and active filters to the Query Handling module.
3.  The query is embedded.
4.  Context Retrieval module searches ChromaDB using the query embedding and the metadata filters.
5.  ChromaDB returns the most relevant text chunks matching the query and filters.
6.  These chunks and the original query form a prompt for GPT-4o.
7.  LLM generates an answer, displayed in the UI with source document references.

**C. Loading Existing Data:**
1.  On application start or when selected, the system attempts to load the persisted ChromaDB collection.
2.  If successful, available metadata filters are populated, and the RAG chain is initialized.

## 5. Tooling and Library Choices

*   **Programming Language:** Python 3.9+
*   **Core RAG Framework:** Langchain, Langchain-Experimental (for SemanticChunker)
*   **LLM:** Azure OpenAI GPT-4o (via `langchain-openai`)
*   **Embeddings:** Azure OpenAI `text-embedding-3-small` (via `langchain-openai`)
*   **Web UI:** Streamlit
*   **Vector Store:** ChromaDB (`chromadb`) for persistent storage and metadata filtering.
*   **Document Loaders:**
    *   PDF: `PyPDFLoader` (Langchain)
    *   DOCX, PPTX, Images (OCR): `UnstructuredWordDocumentLoader`, `UnstructuredPowerPointLoader`, `UnstructuredImageLoader` (Langchain, using `unstructured` library and Tesseract OCR).
    *   TXT: `TextLoader` (Langchain)
*   **Text Chunking:** `SemanticChunker` (Langchain Experimental)
*   **Environment Management:** `python-dotenv`