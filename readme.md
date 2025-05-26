# RAG Chatbot - Technical Challenge

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of uploaded document. It leverages Python, Streamlit for the UI, Langchain for orchestration, Azure OpenAI (GPT-4o and text-embedding-3-small), and ChromaDB for persistent vector storage with metadata filtering.

## Features
*   **Multi-Format Document Ingestion:** Supports PDF, DOCX (Word), PPTX (PowerPoint), TXT, and common image formats (PNG, JPG for OCR).
*   **User-Provided Metadata:** Users can optionally specify Company Name and Year for each uploaded document during ingestion.
*   **Semantic Chunking:** Documents are split into semantically meaningful chunks for improved retrieval relevance.
*   **Persistent Vector Storage:** Processed data (embeddings and metadata) is stored in ChromaDB, allowing for data to persist across sessions.
*   **Metadata-Based Filtering:** Users can filter their queries by Company Name and Year, narrowing the search to specific documents.
*   **Azure OpenAI Integration:** Uses GPT-4o for advanced question answering and `text-embedding-3-small` for generating high-quality text embeddings.
*   **Interactive UI:** Built with Streamlit, providing an easy-to-use interface for file uploads, metadata input, querying, and viewing results.
*   **Source Highlighting:** Displays snippets of source documents used to generate the answer, enhancing transparency.

## Architecture Overview

The system follows a standard RAG pipeline:
1.  **Document Ingestion & Preprocessing:**
    *   Users upload documents and provide optional metadata (Company Name, Year) via the Streamlit interface.
    *   Langchain loaders parse different file formats (including OCR for images).
    *   Text is then chunked using Langchain's `SemanticChunker` with the Azure OpenAI embedding model.
2.  **Embedding & Storage:**
    *   Each semantic chunk is converted into a vector embedding using Azure OpenAI `text-embedding-3-small`.
    *   These embeddings, along with the text chunks and associated metadata (source, company, year), are stored in a persistent ChromaDB collection.
3.  **Retrieval:**
    *   When a user asks a question and selects optional filters (Company, Year):
        *   The query is embedded.
        *   ChromaDB is searched for the most similar chunks, applying the selected metadata filters.
4.  **Generation:**
    *   The retrieved chunks and the user's query are used to construct a prompt for Azure OpenAI GPT-4o.
    *   GPT-4o generates an answer grounded in the provided context.
5.  **User Interface:**
    *   Streamlit handles all user interactions, displays results, and manages data source options (load existing DB or upload new).


## Project Structure
Use code with caution.
Markdown
rag_chatbot_project/
├── requirements.txt
├── app.py
├── core/
│ ├── init.py
│ ├── document_processor.py
│ ├── embedding_handler.py
│ ├── vector_store_handler.py
│ └── llm_handler.py
├── README.md
└── HLD.md
## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   Azure OpenAI account with deployed models for Chat (e.g., GPT-4o) and Embeddings (e.g., `text-embedding-3-small`).
    *   Tesseract OCR engine installed and accessible in your system PATH.
        *   **macOS:** `brew install tesseract tesseract-lang`
        *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng`
        *   **Windows:** Download installer from Tesseract at UB Mannheim, install, and add to PATH.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd rag_chatbot_project
    ```

3.. **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    Create a `.env` file in the root of the project (`rag_chatbot_project/.env`) with your Azure OpenAI credentials:
    ```env
    # Azure OpenAI Credentials
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
    OPENAI_API_VERSION="2024-02-01"
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME="your-text-embedding-3-small-deployment-name"
    ```
    Replace placeholders with your actual details.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Navigate to the project directory.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  **Data Management (Sidebar):**
    *   **Load Existing Data:** On startup, the app attempts to load data from the local ChromaDB store. You can also explicitly reload.
    *   **Upload New Documents:**
        *   Select this mode to upload new files. This will replace the current data in the ChromaDB collection upon processing.
        *   Use the file uploader to select one or more documents (PDF, DOCX, PPTX, TXT, PNG, JPG).
        *   For each uploaded file, optionally provide a "Company Name" and "Year" in the input fields that appear.
        *   Click "Process Uploaded Documents". Processing (especially semantic chunking and embedding) can take time depending on file size and count.
2.  **Filtering (Sidebar):**
    *   If data is loaded, dropdowns for "Company Name" and "Year" will be populated with available metadata.
    *   Select filters to apply them to subsequent queries. The RAG chain will be re-initialized to search only within the filtered context.
3.  **Asking Questions (Main Panel):**
    *   Type your question into the chat input field.
    *   The chatbot will respond based on the (filtered) content from the documents.
    *   You can expand a section to see snippets of the source documents used to generate the answer.
4.  **Deleting Data (Sidebar):**
    *   An option is available to delete the currently stored ChromaDB collection.

## Testing and Evaluation

*   Use the provided Amazon 2023 Annual Report PDF or other sample annual reports.
*   Test with various file types (PDF, DOCX, PPTX, TXT, images with text).
*   Verify that metadata input works and that filtering correctly limits search results.
*   Assess the quality and relevance of answers, and the accuracy of source document attribution.
*   Check how the system handles cases where information is not found in the (filtered) documents.

## Tooling Choices

*   **Python:** Core language.
*   **Streamlit:** Web UI.
*   **Langchain & Langchain-Experimental:** RAG orchestration, document loading, semantic chunking, prompt management, LLM/embedding model integration.
*   **Azure OpenAI Service:** GPT-4o (chat) and `text-embedding-3-small` (embeddings).
*   **ChromaDB:** Persistent vector database with metadata filtering.
*   **Unstructured Library:** Backend for parsing complex file formats (DOCX, PPTX, OCR).
*   **Tesseract OCR:** For text extraction from images.
*   **`python-dotenv`:** Environment variable management.
