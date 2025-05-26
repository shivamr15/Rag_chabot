import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

def get_azure_openai_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        max_tokens=1000
    )

def create_rag_chain(retriever: VectorStoreRetriever, llm: AzureChatOpenAI):


    prompt_template = """You are an AI assistant designed to answer questions based on the provided context from annual reports.
    Use the following pieces of context to answer the question at the end.
    If the context does not contain the answer, state that the information is not found in the provided documents.
    Do not make up information or answer questions outside of the given context.
    Be concise and precise. If possible, cite the source document if available in metadata.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    logger.info("RAG chain created.")
    return qa_chain
def get_answer(chain: RetrievalQA, query: str) -> dict:
    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        logger.error(f"Error getting answer from RAG chain: {e}")
        return {"result": "Sorry, I encountered an error while processing your question.", "source_documents": []}