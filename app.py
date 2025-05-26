import streamlit as st
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any
from core.document_processor import load_documents_from_uploaded_files, chunk_documents
from core.embedding_handler import get_azure_openai_embeddings
from core.vector_store_handler import (create_and_persist_chroma_db, load_chroma_db, delete_chroma_collection, delete_entire_chroma_db_directory, DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME)
from core.llm_handler import get_azure_openai_llm, create_rag_chain, get_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()



st.set_page_config(page_title="RAG Chatbot", layout="wide")
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "messages" not in st.session_state: st.session_state.messages = []
if "rag_ready" not in st.session_state: st.session_state.rag_ready = False
if "data_source_mode" not in st.session_state: st.session_state.data_source_mode = "load_existing"
if "loaded_from_disk_successfully" not in st.session_state: st.session_state.loaded_from_disk_successfully = False
if "initial_load_attempted" not in st.session_state: st.session_state.initial_load_attempted = False
if "available_companies" not in st.session_state: st.session_state.available_companies = []
if "available_years" not in st.session_state: st.session_state.available_years = []
if "selected_company_filter" not in st.session_state: st.session_state.selected_company_filter = "All"
if "selected_year_filter" not in st.session_state: st.session_state.selected_year_filter = "All"
if "file_metadata_inputs" not in st.session_state: st.session_state.file_metadata_inputs = {}


def get_available_metadata_filters(vector_store):
    if not vector_store or vector_store._collection.count() == 0:
        return [], []
    try:
        results = vector_store._collection.get(include=["metadatas"])
        metadatas = results.get('metadatas', [])
        if not metadatas: return [], []
        all_companies, all_years = set(), set()
        for meta in metadatas:
            if meta:
                company = meta.get("company_name")
                year = meta.get("year")
                if company and company not in ["Unknown", "Not Specified"]: all_companies.add(company)
                if year and year not in ["Unknown", "Not Specified"]: all_years.add(str(year))
        return sorted(list(all_companies)), sorted(list(all_years), reverse=True)
    except Exception as e:
        logger.error(f"Error fetching metadata for filters: {e}", exc_info=True)
        return [], []

def _initialize_rag_chain_from_vs(vector_store, company_filter="All", year_filter="All"):
    if not vector_store or vector_store._collection.count() == 0:
        st.session_state.rag_chain = None
        st.session_state.rag_ready = False
        return False
    try:
        llm = get_azure_openai_llm()
        search_kwargs = {"k": 5}
        filter_dict = {}
        if company_filter and company_filter != "All": filter_dict["company_name"] = company_filter
        if year_filter and year_filter != "All": filter_dict["year"] = str(year_filter)
        if filter_dict: search_kwargs["filter"] = filter_dict
        logger.info(f"Retriever search_kwargs: {search_kwargs}")
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        st.session_state.rag_chain = create_rag_chain(retriever, llm)
        st.session_state.rag_ready = True
        logger.info("RAG chain (re)initialized successfully with current filters.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}", exc_info=True)
        st.error(f"Error initializing RAG system: {e}")
        st.session_state.rag_chain = None; st.session_state.rag_ready = False
        return False

def process_and_store_documents(uploaded_files_with_metadata: List[Dict[str, Any]]):
    if not uploaded_files_with_metadata:
        st.error("Please upload at least one document and provide its metadata to process.")
        return False
    with st.spinner("Processing documents... This may take a while. Please wait."):
        try:
            logger.info("Starting new document processing with user-provided metadata...")
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            st.session_state.rag_ready = False
            st.session_state.loaded_from_disk_successfully = False
            delete_chroma_collection(DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME) 
            raw_documents = load_documents_from_uploaded_files(uploaded_files_with_metadata)
            if not raw_documents:
                st.error("No text could be extracted. Please check files.")
                return False
            embeddings_model = get_azure_openai_embeddings()
            text_chunks = chunk_documents(raw_documents, embeddings=embeddings_model) 
            if not text_chunks:
                st.error("Failed to chunk documents.")
                return False
            st.session_state.vector_store = create_and_persist_chroma_db(
                text_chunks, embeddings_model, DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME
            )
            if st.session_state.vector_store:
                logger.info("New ChromaDB collection created/updated and persisted.")
                st.session_state.messages = [] 
                st.session_state.available_companies, st.session_state.available_years = get_available_metadata_filters(st.session_state.vector_store)
                return _initialize_rag_chain_from_vs(st.session_state.vector_store, st.session_state.selected_company_filter, st.session_state.selected_year_filter)
            else:
                st.error("Failed to create/persist ChromaDB. Check logs and Azure credentials.")
                return False
        except Exception as e:
            logger.error(f"Error during new document processing: {e}", exc_info=True)
            st.error(f"An critical error occurred: {e}")
            st.session_state.rag_ready = False
            return False

def attempt_load_and_initialize_existing_data():
    logger.info(f"Attempting to load existing data from ChromaDB: {DEFAULT_CHROMA_DB_PATH}, collection: {DEFAULT_COLLECTION_NAME}...")
    st.session_state.loaded_from_disk_successfully = False
    st.session_state.rag_ready = False
    if not os.path.exists(DEFAULT_CHROMA_DB_PATH): return False
    with st.spinner("Loading existing data..."):
        try:
            embeddings_model = get_azure_openai_embeddings()
            st.session_state.vector_store = load_chroma_db(embeddings_model, DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME)
            if st.session_state.vector_store and st.session_state.vector_store._collection.count() > 0:
                st.session_state.available_companies, st.session_state.available_years = get_available_metadata_filters(st.session_state.vector_store)
                if _initialize_rag_chain_from_vs(st.session_state.vector_store, st.session_state.selected_company_filter, st.session_state.selected_year_filter):
                    st.session_state.loaded_from_disk_successfully = True
                    logger.info("Existing ChromaDB data loaded and RAG chain initialized.")
                    return True
            elif st.session_state.vector_store and st.session_state.vector_store._collection.count() == 0:
                 logger.warning("ChromaDB loaded but collection is empty."); st.session_state.vector_store = None
            else: st.warning("Could not load existing ChromaDB data.")
        except Exception as e:
            logger.error(f"Error attempting to load existing ChromaDB data: {e}", exc_info=True)
            st.error(f"Failed to load existing data: {e}.")
    return False

if not st.session_state.initial_load_attempted:
    logger.info("Performing initial attempt to load existing data from ChromaDB...")
    attempt_load_and_initialize_existing_data()
    st.session_state.initial_load_attempted = True

st.title("RAG Chatbot")

with st.sidebar:
    st.header("Data Management")
    current_mode_index = 0 if st.session_state.data_source_mode == "load_existing" else 1
    if not st.session_state.loaded_from_disk_successfully and st.session_state.initial_load_attempted:
        current_mode_index = 1
    chosen_mode_option = st.radio(
        "Choose an action:",
        ("Chat with Existing Data", "Upload New Documents (replaces existing collection)"),
        index=current_mode_index, key="data_mode_radio_user_meta"
    )
    if chosen_mode_option == "Chat with Existing Data" and st.session_state.data_source_mode != "load_existing":
        st.session_state.data_source_mode = "load_existing"
        if not st.session_state.loaded_from_disk_successfully: attempt_load_and_initialize_existing_data()
        st.rerun()
    elif chosen_mode_option == "Upload New Documents (replaces existing collection)" and st.session_state.data_source_mode != "upload_new":
        st.session_state.data_source_mode = "upload_new"
        st.session_state.rag_ready = False
        st.rerun()
    if st.session_state.data_source_mode == "upload_new":
        st.subheader("Upload & Process New Data")
        uploaded_files = st.file_uploader(
            "Upload report files (PDF, DOCX, PPTX, TXT, PNG, JPG).",
            type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True, key="file_uploader_user_meta"
        )
        if uploaded_files:
            st.markdown("**Provide Optional Metadata for Each File:**")
            for i, up_file in enumerate(uploaded_files):
                file_key = up_file.name  
                if file_key not in st.session_state.file_metadata_inputs:
                    st.session_state.file_metadata_inputs[file_key] = {"company_name": "", "year": ""}
                with st.expander(f"Metadata for: {up_file.name}", expanded=(i==0)):
                    st.session_state.file_metadata_inputs[file_key]["company_name"] = st.text_input(
                        f"Company Name (for {up_file.name})", 
                        value=st.session_state.file_metadata_inputs[file_key]["company_name"],
                        key=f"company_name_{file_key}"
                    )
                    st.session_state.file_metadata_inputs[file_key]["year"] = st.text_input(
                        f"Year (for {up_file.name})",
                        value=st.session_state.file_metadata_inputs[file_key]["year"],
                        key=f"year_{file_key}",
                        placeholder="e.g., 2023"
                    )
        current_uploaded_file_ids = {f.name for f in uploaded_files} if uploaded_files else set()
        keys_to_delete = [k for k in st.session_state.file_metadata_inputs if k not in current_uploaded_file_ids]
        for k in keys_to_delete:
            del st.session_state.file_metadata_inputs[k]
        if st.button("Process Uploaded Documents", key="process_user_meta_docs", type="primary", use_container_width=True):
            if uploaded_files:
                files_with_user_metadata = []
                for up_file in uploaded_files:
                    file_key = up_file.name
                    meta = st.session_state.file_metadata_inputs.get(file_key, {"company_name": "", "year": ""})
                    files_with_user_metadata.append({
                        "file": up_file,
                        "company_name": meta["company_name"].strip(),
                        "year": meta["year"].strip()
                    })
                if process_and_store_documents(files_with_user_metadata): 
                    st.success("Documents processed and saved to ChromaDB!")
                    st.session_state.file_metadata_inputs = {}
            else:
                st.warning("Please upload documents first.")
    elif st.session_state.data_source_mode == "load_existing":
        st.subheader("Existing Data Status")
        if st.session_state.loaded_from_disk_successfully and st.session_state.rag_ready:
            st.success("✅ ChromaDB loaded. Ready for chat.")
        elif os.path.exists(DEFAULT_CHROMA_DB_PATH):
            st.warning("ChromaDB data found, but not active for chat.")
            if st.button("Reload Existing Data", key="reload_chroma_user_meta", use_container_width=True):
                if attempt_load_and_initialize_existing_data(): st.success("ChromaDB reloaded!")
                else: st.error("Failed to reload.")
                st.rerun()
        else: st.info("No ChromaDB data found. Switch to 'Upload New Documents'.")

    st.header("Filters (for existing data)")
    if st.session_state.vector_store and st.session_state.vector_store._collection.count() > 0:
        company_options = ["All"] + st.session_state.available_companies
        year_options = ["All"] + st.session_state.available_years
        company_idx = company_options.index(st.session_state.selected_company_filter) if st.session_state.selected_company_filter in company_options else 0
        year_idx = year_options.index(st.session_state.selected_year_filter) if st.session_state.selected_year_filter in year_options else 0
        new_company_filter = st.selectbox("Filter by Company:", company_options, index=company_idx, key="company_filter_sb_user")
        new_year_filter = st.selectbox("Filter by Year:", year_options, index=year_idx, key="year_filter_sb_user")
        if new_company_filter != st.session_state.selected_company_filter or new_year_filter != st.session_state.selected_year_filter:
            st.session_state.selected_company_filter = new_company_filter
            st.session_state.selected_year_filter = new_year_filter
            logger.info(f"Filters changed. Re-initializing RAG chain.")
            _initialize_rag_chain_from_vs(st.session_state.vector_store, new_company_filter, new_year_filter)
            st.session_state.messages = []
            st.rerun()
    else: st.caption("Load or process data to enable filters.")
    if os.path.exists(DEFAULT_CHROMA_DB_PATH):
        if st.button("Delete Stored Collection", key="delete_chroma_coll_user_meta", use_container_width=True):
            with st.spinner("Deleting ChromaDB collection..."):
                delete_chroma_collection(DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME)
                st.session_state.vector_store = None; st.session_state.rag_chain = None
                st.session_state.rag_ready = False; st.session_state.loaded_from_disk_successfully = False
                st.session_state.messages = []; st.session_state.available_companies, st.session_state.available_years = [], []
                st.session_state.data_source_mode = "upload_new" 
            st.success(f"ChromaDB collection '{DEFAULT_COLLECTION_NAME}' deleted.")
            st.rerun()

if st.session_state.rag_ready and st.session_state.rag_chain:
    filter_info = []
    if st.session_state.selected_company_filter != "All": filter_info.append(f"Co: {st.session_state.selected_company_filter}")
    if st.session_state.selected_year_filter != "All": filter_info.append(f"Year: {st.session_state.selected_year_filter}")
    filter_text = f" (Filters: {', '.join(filter_info)})" if filter_info else " (No filters)"
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("AI is thinking..."):
                try:
                    response = get_answer(st.session_state.rag_chain, prompt)
                    full_response_text = response.get("result", "Sorry, I could not find an answer.")
                    source_docs = response.get("source_documents")
                    if source_docs:
                        with st.expander("View relevant context from documents"):
                            for i, doc in enumerate(source_docs):
                                meta = doc.metadata
                                src = meta.get('source', 'N/A'); co = meta.get('company_name', 'N/A'); yr = meta.get('year', 'N/A')
                                if len(src) > 30 : src = "..." + src[-27:]
                                st.write(f"**Context {i+1}** (Src: {src}, Co: {co}, Yr: {yr})")
                                st.caption(doc.page_content[:350] + "...") 
                except Exception as e:
                    logger.error(f"Error during query: {e}", exc_info=True)
                    full_response_text = f"An error occurred: {str(e)}"
            message_placeholder.markdown(full_response_text)
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})
elif st.session_state.data_source_mode == "upload_new" and not st.session_state.rag_ready :
    st.info("Please upload documents and provide metadata using the sidebar to begin chatting.")
elif st.session_state.data_source_mode == "load_existing" and not st.session_state.rag_ready:
    if os.path.exists(DEFAULT_CHROMA_DB_PATH): st.info("ChromaDB data found. Try 'Reload Existing Data'.")
    else: st.info("No ChromaDB data. Switch to 'Upload New Documents'.")
else: st.info("System initializing or waiting for data. Select an option from the sidebar.")