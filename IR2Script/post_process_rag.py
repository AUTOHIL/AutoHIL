import os
from dotenv import load_dotenv
from typing import List
import logging

# --- Core LangChain components ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Data loading (dynamic loader) ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)

# --- 2. Text splitting ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 3. Vector embeddings (!!! Fix: import Google Gemini !!!) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 4. Vector store (ChromaDB) ---
from langchain_community.vectorstores import Chroma

# --- 5. LLM ---
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='./IR2Script/log/postProcess_rag.log'
)


POST_PROCESS_DB = "./rag_db/post_processing_db"
POST_PROCESS_COLLECTION = "post_process_collection"

# ===================================================================
# Helper functions
# ===================================================================

def load_document(file_path: str) -> list[Document]:
    """(Helper) Dynamically select and load a single file based on its extension."""
    _, extension = os.path.splitext(file_path)
    
    logging.info(f"  > Loading: {file_path} (type: {extension})")

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".json":
            loader = JSONLoader(
                file_path=file_path, 
                jq_schema='.[]', 
                text_content=True
            )
        elif extension in [".txt", ".py", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            logging.info(f"  > Warning: skipping unsupported file type {extension} ({file_path})")
            return []
        
        return loader.load()

    except Exception as e:
        logging.info(f"  > Error while loading file {file_path}: {e}")
        return []

def get_gemini_llm():
    api_key = os.getenv("GOOGLE_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set (for LLM)")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key
    )

    return llm

def get_google_embeddings():

    # Check GOOGLE_API_KEY
    google_api_key = os.getenv("GOOGLE_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set (for Embedding)")
    
    # GoogleGenerativeAIEmbeddings will automatically read the key from the GOOGLE_API_KEY env var
    # We use 'text-embedding-004'
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    return embeddings

def format_docs(docs: list[Document]) -> str:
    """Merge a list of retrieved Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def extract_py_code(content):
    start_marker = "```python"
    if start_marker in content:
        start_index = content.find(start_marker) + len(start_marker)

        end_marker = "```"
        end_index = content.rfind(end_marker)

        py_content = content[start_index:end_index ]
    else:
        py_content = content

    return py_content


def create_post_process_index(file_paths: List[str]):
    """
    Load, split, embed files, and save the index to disk.
    """
    logging.info("\n[Stage 1: Index processing]")
    
    logging.info("  1. Start loading the specified file list...")
    all_docs = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_docs.extend(docs)
        else:
            logging.error(f"  > Warning: skipping non-existent file: {file_path}")
    
    if not all_docs:
        logging.error("No valid documents were loaded; terminating the RAG pipeline.")
        return
    logging.info(f"\n  > Document loading complete, total {len(all_docs)} documents.")

    logging.info("  2. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    logging.info(f"    Number of document chunks (Splits) after splitting: {len(splits)}")
    
    if not splits:
        logging.error("No content after text splitting; terminating the RAG pipeline.")
        return

    # --- 3. Vector embeddings
    logging.info("  3. Initializing embedding model (Google Gemini API)...")
    embeddings = get_google_embeddings()

    logging.info("  4. Creating ChromaDB vector index (in-memory mode)...")
    # from_documents will batch-compute vectors using the Google embeddings configured above
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=POST_PROCESS_DB,
        collection_name=POST_PROCESS_COLLECTION
    )
    
    logging.info(f"--- Index creation completed and saved to {POST_PROCESS_DB} ---")


# ===================================================================
# RAG pipeline main function
# ===================================================================

def post_processing(req_id, user_json, user_query, adjusted_file):
    """
    Load the on-disk vector database and execute the RAG query.
    """
    logging.info("\n[---------------------- Code post-processing --------------------]")

    # 1. Check whether the database exists
    if not os.path.exists(POST_PROCESS_DB):
        logging.error(f"Error: vector database folder does not exist; please create the index first.")
        return
    
    # 2. Initialize embeddings
    embeddings = get_google_embeddings()

    # 3. Load the database
    vectorstore = Chroma(
        persist_directory=POST_PROCESS_DB,
        embedding_function=embeddings,
        collection_name=POST_PROCESS_COLLECTION
    )

    # 4. Initialize LLM
    llm = get_gemini_llm()
    
    # 5. RAG chain
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 550
        }
    )
    doc_file1 = retriever.invoke(user_query)

    template = """
The following Python test script was generated from a Behavior Tree JSON via rule-based programming. Please correct specific detail errors missed during the conversion—such as missing intermediate variable definitions or unquoted strings—without altering the script's structure.
Ensure the code is syntactically and semantically correct (executable without errors). Additionally, validate and enforce the corresponding API parameter constraints based on the provided API knowledge base.
Output the corrected code directly. Do not provide any explanation.

[api knowledge base]
{context}

[behavior tree json]
{json}

[code]
{question}

Your Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    context = format_docs(doc_file1)

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = extract_py_code(rag_chain.invoke({
        "context": context,
        "json":user_json,
        "question": user_query
        }))

    logging.info(f"\n[Requirement ID]: {req_id}")
    logging.info(f"\n[Post-processed code]: {answer}")
    logging.info("\n------------------------ Post-processing pipeline finished -----------------------")
    
    retrieved_docs = retriever.invoke(user_query)
    
    with open(adjusted_file, 'w', encoding='utf-8') as f1:
        f1.write(answer)
    
    return answer
