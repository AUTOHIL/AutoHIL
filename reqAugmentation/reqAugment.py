import os
from dotenv import load_dotenv
from typing import List
import logging
import json

# --- Core LangChain components ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Document loaders (dynamic loader) ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)

# --- 2. Text splitting ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 3. Vector embeddings (Google Gemini) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 4. Vector store (ChromaDB) ---
from langchain_community.vectorstores import Chroma

# --- 5. LLM ---
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

REQ_AUGMENT_DB = "./rag_db/augment_db"
REQ_AUGMENT_COLLECTION = "req_augment_collection"

# ===================================================================
# Helper functions
# ===================================================================

def format_requirement_dict(req_dict) -> str:
    """
    Convert a single requirement dict into a JSON string, preserving all original fields.
    """
    return json.dumps(req_dict, indent=2, ensure_ascii=False)

def load_document(file_path: str) -> list[Document]:
    """(helper) Dynamically choose and load a single file based on its extension."""
    _, extension = os.path.splitext(file_path)
    
    logging.info(f"  > Loading: {file_path} (type: {extension})")

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                # if the JSON file is not a list, load it as a single Document
                return [Document(page_content=json.dumps(data, ensure_ascii=False))]
                
            # format each dict in the list as a string
            formatted_reqs = [format_requirement_dict(d) for d in data if isinstance(d, dict)]
            
            # 1. Merge the entire list into one large Document (suitable for small files)
            full_content = "\n\n".join(formatted_reqs)
            
            # Document metadata can include basic file information
            metadata = {"source": file_path, "total_requirements": len(formatted_reqs)}
            
            return [Document(page_content=full_content, metadata=metadata)]
        elif extension in [".txt", ".py", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            logging.warning(f"  > Warning: skipping unsupported file type {extension} ({file_path})")
            return []
        
        return loader.load()

    except Exception as e:
        logging.error(f"  > Error while loading file {file_path}: {e}")
        return []

def get_deepseek_llm():
    """(LLM) Load and return a DeepSeek LLM instance."""
    api_key = os.getenv("DS_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set (for LLM)")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    return llm

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

    # check GOOGLE_API_KEY
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

def create_reqAug_index(file_paths: List[str]):
    all_docs = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_docs.extend(docs)
        else:
            logging.info(f"Path is not a file, skipping: {file_path}")
    
    if not all_docs:
        logging.info("No documents were loaded; cannot create index.")
        return
    logging.info(f"Loaded {len(all_docs)} documents in total.")
    
    # Document splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"]
    )
    splits = text_splitter.split_documents(all_docs)
    logging.info(f"After text splitting, got {len(splits)} chunks.")
    
    if not splits:
        logging.info("No chunks after text splitting; cannot create index.")
        return
    
    embeddings = get_google_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=REQ_AUGMENT_DB,
        collection_name=REQ_AUGMENT_COLLECTION
    )
    logging.info(f"Vector database created and saved to: {REQ_AUGMENT_DB}")
    
def req_augment(req_id, req_org, knowledge_files):
    # logging.info("\n[-------------------------- Requirement augmentation --------------------------]")
    if not os.path.exists(REQ_AUGMENT_DB):
        logging.error("Vector database does not exist; please create the index first.")
        return
    
    embeddings = get_google_embeddings()
    vectorstore = Chroma(
        persist_directory=REQ_AUGMENT_DB,
        embedding_function=embeddings,
        collection_name=REQ_AUGMENT_COLLECTION
    )
    
    llm = get_gemini_llm()
    
    # retriever1: req point corpus
    retriever1 = vectorstore.as_retriever(search_kwargs={
        "k": 50,
        "filter": {"source": knowledge_files[0]}
    })
    # retriever2: module config file
    retriever2 = vectorstore.as_retriever(search_kwargs={
        "k": 20,
        "filter": {"source": knowledge_files[1]}
    })
    doc_file1 = retriever1.invoke(req_org)
    doc_file2 = retriever2.invoke(req_org)
    all_docs = doc_file1 + doc_file2
    logging.info(f"Retrieved related document chunks: {len(all_docs)}")
    
    template = """[Role] You are a professional Expert in Automotive Domain Requirement Analysis, specializing in writing precise, complete, and unambiguous requirements.
[Task] Please assist me in performing a Requirement Augmentation Task based on the raw requirements I provide. You must adhere to the following rules:
1. Intent Preservation & Expansion: You are strictly prohibited from modifying the original intent of the requirement. Instead, perform appropriate expansion: Search the requirement corpus [You generated above] for test points and requirement points that align with the current scenario. Rewrite the raw requirement into a more complete form containing a clear Goal, Input/Trigger Conditions, Expected Output, and Pass/Fail Criteria.
2. Parameter Integration: Consult the Config file to fill in corresponding configurations and parameter values, such as DTC codes, Message IDs, cycle times, thresholds, etc. Ensure the correct parameters are selected and strictly avoid confusion.
3. Advanced Testing Logic: Incorporate hysteresis effects and gradient/ramp testing (where applicable). When defining verification methods, prioritize recording logs and analyzing historical performance from the logs rather than simply reading real-time current values.
4. Output Format: Output the result as a complete, cohesive text paragraph. Output ONLY the enhanced requirement description.

[Original Requirement] 
{req_org}

[requirement corpus]
{req_corpus}

[Config File]
{cfg_file}

Your answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    req_corpus = format_docs(doc_file1)
    cfg_file = format_docs(doc_file2)
    
    chain = (prompt | llm | StrOutputParser())
    answer = chain.invoke({
        "req_org": req_org,
        "req_corpus": req_corpus,
        "cfg_file": cfg_file
    })
    
    logging.info(f"\n[Requirement ID]: {req_id}\n[Augmented requirement]:\n{answer}\n")
    logging.info("\n-------------------------- Requirement augmentation pipeline finished --------------------------]")
    return answer
