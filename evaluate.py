import time
import os
import logging
from dotenv import load_dotenv
from core.embedding_handler import get_azure_openai_embeddings
from core.vector_store_handler import load_chroma_db, DEFAULT_CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME
from core.llm_handler import get_azure_openai_llm, create_rag_chain, get_answer
import json

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("RAG_EVALUATION")

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What were the total net sales reported by Amazon in 2023?",
        "company_filter": "All", 
        "year_filter": "All",
        "expected_keywords_in_answer": ["billion", "net sales", "Amazon"], 
        "expected_keywords_in_context": ["net sales", "revenue", "2023"]
    },
    {
        "id": 2,
        "question": "What are some key risks mentioned in the Amazon 2023 annual report?",
        "company_filter": "All",
        "year_filter": "All",
        "expected_keywords_in_answer": ["risk", "competition", "factors"],
        "expected_keywords_in_context": ["risk factors", "competition", "challenges"]
    },
    {
        "id": 3,
        "question": "Who is the current CEO of amazon?", 
        "company_filter": "amazon", 
        "year_filter": "All", 
        "expected_keywords_in_answer": ["Satya Nadella", "CEO", "amazon"],
        "expected_keywords_in_context": ["chief executive officer", "Andrew R. Jassy"]
    },

    {
        "id": 4,
        "question": "What is the price of a Venti Latte at Starbucks in their latest report?",
        "company_filter": "Starbucks", 
        "year_filter": "All",
        "expected_keywords_in_answer": ["not found", "not specified", "unable to determine"], 
        "expected_keywords_in_context": [] 
    }
]

def run_evaluation():
    try:
        embeddings_model = get_azure_openai_embeddings()
        llm = get_azure_openai_llm()
        vector_store = load_chroma_db(
            embeddings=embeddings_model,
            db_path=DEFAULT_CHROMA_DB_PATH,
            collection_name=DEFAULT_COLLECTION_NAME
        )
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}", exc_info=True)
        return

    if not vector_store or vector_store._collection.count() == 0:
        logger.error(f"Vector store at '{DEFAULT_CHROMA_DB_PATH}' (collection: '{DEFAULT_COLLECTION_NAME}') is not loaded or is empty. Cannot run evaluation.")
        logger.info("Please ensure you have processed documents using app.py first.")
        return
    
    logger.info("Components initialized successfully.")

    total_latency = 0
    num_questions = len(TEST_QUESTIONS)
    successful_runs = 0

    evaluation_results = []

    for i, item in enumerate(TEST_QUESTIONS):
        question_id = item["id"]
        query = item["question"]
        company_filter = item.get("company_filter", "All")
        year_filter = item.get("year_filter", "All")
        expected_keywords_answer = item.get("expected_keywords_in_answer", [])
        expected_keywords_context = item.get("expected_keywords_in_context", [])


        logger.info(f"\n--- Evaluating Question ID: {question_id} ---")
        logger.info(f"Query: {query}")
        logger.info(f"Filters: Company='{company_filter}', Year='{year_filter}'")


        search_kwargs = {"k": 3}
        filter_dict = {}
        if company_filter != "All":
            filter_dict["company_name"] = company_filter
        if year_filter != "All":
            filter_dict["year"] = str(year_filter)
        
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to create retriever for Q_ID {question_id}: {e}", exc_info=True)
            evaluation_results.append({"id": question_id, "status": "Retriever Error", "error": str(e)})
            continue
        retrieval_start_time = time.time()
        try:
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error during retrieval for Q_ID {question_id}: {e}", exc_info=True)
            evaluation_results.append({"id": question_id, "status": "Retrieval Error", "error": str(e)})
            continue
        retrieval_latency = time.time() - retrieval_start_time
        logger.info(f"Retrieval Latency: {retrieval_latency:.4f} seconds")

        logger.info(f"Retrieved {len(retrieved_docs)} documents for context.")
        combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        for idx, doc in enumerate(retrieved_docs):
            logger.debug(f"  CONTEXT CHUNK {idx+1} (Source: {doc.metadata.get('source', 'N/A')} | Co: {doc.metadata.get('company_name','N/A')} | Yr: {doc.metadata.get('year','N/A')} ):")
            logger.debug(f"  >>> {doc.page_content[:300]}...") 

        context_keywords_found = [kw for kw in expected_keywords_context if kw.lower() in combined_context.lower()]
        context_relevance_score = len(context_keywords_found) / len(expected_keywords_context) if expected_keywords_context else 1.0
        logger.info(f"Basic Context Keyword Match: {context_relevance_score*100:.2f}% ({len(context_keywords_found)}/{len(expected_keywords_context)} keywords found)")

        try:
            qa_chain = create_rag_chain(retriever, llm) 
        except Exception as e:
            logger.error(f"Failed to create RAG chain for Q_ID {question_id}: {e}", exc_info=True)
            evaluation_results.append({"id": question_id, "status": "Chain Creation Error", "error": str(e)})
            continue
        llm_start_time = time.time()
        try:
            response = get_answer(qa_chain, query) 
            llm_answer = response.get("result", "Error: No result in response")
            source_documents_from_chain = response.get("source_documents", [])
        except Exception as e:
            logger.error(f"Error during LLM generation for Q_ID {question_id}: {e}", exc_info=True)
            evaluation_results.append({"id": question_id, "status": "LLM Generation Error", "error": str(e)})
            continue
        llm_latency = time.time() - llm_start_time
        logger.info(f"LLM Generation Latency: {llm_latency:.4f} seconds")

        end_to_end_question_latency = retrieval_latency + llm_latency
        total_latency += end_to_end_question_latency
        
        logger.info(f"End-to-End Latency for this question: {end_to_end_question_latency:.4f} seconds")
        logger.info(f"LLM Answer: {llm_answer}")
        
        answer_keywords_found = [kw for kw in expected_keywords_answer if kw.lower() in llm_answer.lower()]
        answer_keyword_score = len(answer_keywords_found) / len(expected_keywords_answer) if expected_keywords_answer else 1.0
        
        logger.info(f"Basic Answer Keyword Match: {answer_keyword_score*100:.2f}% ({len(answer_keywords_found)}/{len(expected_keywords_answer)} keywords found)")
        
        evaluation_results.append({
            "id": question_id,
            "question": query,
            "filters": filter_dict if filter_dict else "None",
            "status": "Success",
            "retrieval_latency_sec": round(retrieval_latency, 4),
            "llm_latency_sec": round(llm_latency, 4),
            "e2e_latency_sec": round(end_to_end_question_latency, 4),
            "retrieved_chunks_count": len(retrieved_docs),
            "context_keyword_score": round(context_relevance_score, 2),
            "answer_keyword_score": round(answer_keyword_score, 2),
            "llm_answer": llm_answer,
            "retrieved_contexts_snippets": [
                {
                    "content": doc.page_content[:200]+"...", 
                    "metadata": doc.metadata
                } for doc in retrieved_docs
            ]
        })
        successful_runs += 1
    logger.info("\nEvaluation Summary -----------")
    if num_questions > 0:
        avg_latency = total_latency / successful_runs if successful_runs > 0 else 0
        logger.info(f"Total Questions Evaluated: {num_questions}")
        logger.info(f"Successful Runs: {successful_runs}")
        logger.info(f"Average End-to-End Latency (for successful runs): {avg_latency:.4f} seconds")
    else:
        logger.info("No questions were evaluated.")
    logger.info("\n--- Detailed Results ---")
    for res in evaluation_results:
        logger.info(f"Q_ID: {res['id']} | Status: {res['status']}")
        if res['status'] == 'Success':
            logger.info(f"  Question: {res['question']}")
            logger.info(f"  Filters: {res['filters']}")
            logger.info(f"  E2E Latency: {res['e2e_latency_sec']}s (Retrieval: {res['retrieval_latency_sec']}s, LLM: {res['llm_latency_sec']}s)")
            logger.info(f"  Context Score: {res['context_keyword_score']*100:.0f}% | Answer Score: {res['answer_keyword_score']*100:.0f}%")
            logger.info(f"  Answer: {res['llm_answer']}")
            logger.info(f"  Retrieved context snippets ({res['retrieved_chunks_count']}):")
            for i, snip in enumerate(res['retrieved_contexts_snippets']):
                 logger.info(f"    [{i+1}] Meta: {snip['metadata'].get('source','?')} (Co: {snip['metadata'].get('company_name','?')}, Yr: {snip['metadata'].get('year','?')}) | Content: {snip['content']}")
        else:
            logger.error(f"  Error: {res.get('error')}")
        logger.info("-" * 30)

    with open("evaluation_report.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info("Detailed evaluation report saved to evaluation_report.json")

if __name__ == "__main__":
    if not all([os.getenv("AZURE_OPENAI_API_KEY"), 
                os.getenv("AZURE_OPENAI_ENDPOINT"), 
                os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")]):
        logger.error("Azure OpenAI environment variables are not fully set. Please check your .env file.")
    else:
        run_evaluation()