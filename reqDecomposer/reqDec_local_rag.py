import os
from dotenv import load_dotenv
from typing import List
import logging

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

# load environment variables from .env
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # filename='/home/gsc/HIL/reqDecomposer/log/reqDec_rag.log'
)


REQ_DECOMPOSE_DB = "/home/gsc/HIL/rag_db/decomposer_db"
REQ_DECOMPOSE_COLLECTION = "req_decompose_collection"

# ===================================================================
# Helper functions
# ===================================================================

def load_document(file_path: str) -> list[Document]:
    """(helper) Dynamically choose and load a single file based on its extension."""
    _, extension = os.path.splitext(file_path)

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

    google_api_key = os.getenv("GOOGLE_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set (for Embedding)")
    
    # GoogleGenerativeAIEmbeddings will read the key from GOOGLE_API_KEY
    # using model 'text-embedding-004'
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    return embeddings

def format_docs(docs: list[Document]) -> str:
    """Merge a list of retrieved Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_api_pool_index(file_paths: List[str]):
    """
    Load, split, embed files and persist the index to disk.
    """
    
    logging.info("  1. Start loading the specified file list...")
    all_docs = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_docs.extend(docs)
        else:
            logging.warning(f"  > Warning: skipping non-existent file: {file_path}")
    
    if not all_docs:
        logging.error("No valid documents were loaded; terminating the RAG pipeline.")
        return

    logging.info("  2. Text splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    logging.info(f"    Number of document chunks (Splits) after splitting: {len(splits)}")
    
    if not splits:
        logging.error("No content after text splitting; terminating the RAG pipeline.")
        return

    # --- 3. Vector embedding
    logging.info("  3. Initializing embedding model (Google Gemini API)...")
    embeddings = get_google_embeddings()

    logging.info("  4. Creating ChromaDB vector index (in-memory mode)...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=REQ_DECOMPOSE_DB,
        collection_name=REQ_DECOMPOSE_COLLECTION
    )
    
    logging.info(f"--- Index creation completed and saved to {REQ_DECOMPOSE_DB} ---")


# ===================================================================
# RAG flow main function
# ===================================================================

def query_rag(req_id, user_query, knowledge_files):
    """
    Load the vector DB from disk and perform a RAG query.
    """
    # 1. Check whether the database exists
    if not os.path.exists(REQ_DECOMPOSE_DB):
        logging.error(f"Error: vector database folder does not exist; please create the index first.")
        return
    
    # 2. Initialize embeddings
    embeddings = get_google_embeddings()

    # 3. Load the database
    vectorstore = Chroma(
        persist_directory=REQ_DECOMPOSE_DB,
        embedding_function=embeddings,
        collection_name=REQ_DECOMPOSE_COLLECTION
    )

    # 4. Initialize LLM
    llm = get_gemini_llm()
    
    # 5. RAG chain
    retriever1 = vectorstore.as_retriever(
        search_kwargs={
            "k": 550,
            "filter": {"source": knowledge_files[0]}
        }
    )
    retriever2 = vectorstore.as_retriever(
        search_kwargs={
            "k": 20,
            "filter": {"source": knowledge_files[1]}
        }
    )
    doc_file1 = retriever1.invoke(user_query)
    doc_file2 = retriever2.invoke(user_query)
    all_docs = doc_file1 + doc_file2
    logging.info(f"After merging retrievals, total {len(all_docs)} document chunks")

    template = """
[Role] You are an expert in Automotive ECU Functional Test Requirement Analysis. For every functional requirement, test scenarios must be simulated to verify if the function operates correctly under those conditions. The Device Under Test (DUT) is the automotive ECU.
[Task] Please strictly search the API knowledge base I have uploaded, identifying and selecting the correct APIs to break down the requirement description into a series of executable steps. Each step must correspond to a found API, a condition, or a jump instruction, adhering to the following requirements:
1. Closed-Loop Testing: The test must form a complete closed loop, including: Prerequisite configuration. Confirmation that the configuration has taken effect. Process verification (specific implementation of the Action requirement). State transitions (ECU Power On/Off). Comparison with expected results (mandatory; if no comparison is needed, skip this). Restoration to a normal state upon test completion (e.g., clearing DTCs).
Note: Process verification may involve repeated testing. Manual testing or undefined custom test steps are strictly prohibited.
2. API Selection & Constraints: Every atomic operation API must be selected strictly and exclusively from the knowledge base I uploaded. You must strictly select APIs for which parameters are explicitly defined in the requirement; do not select APIs if the input parameters are undefined. Parameter values must not be abstract natural language concepts. Prioritize APIs that interact with external ECUs to retrieve information.
Read/Write operation APIs (get/set) must involve interaction with the ECU. DTC retrieval and clearing operations must interact with the external ECU to take effect.
Explicitly state inputs and outputs in the format: input=[], output=[].
3. Control Flow: Mimic machine code conditions/jumps (e.g., jmp) using natural language to describe loops or jumps (e.g., "Repeat steps 2-8").
[Output] Present the complete test step sequence in a table format. The table must include: Step No., Atomic Operation API, Input, Output, Atomic Step Description, and Expected Result. If loops involve custom counters or lists, please provide necessary variable definitions before the control flow description.
Note: Output only the table, the control structure of the requirement, and any necessary variable definitions. 

[Context]
{context}

[Requirement Item]
{question}

Your answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    context = format_docs(all_docs)
    
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({
        "context": context,
        "question": user_query
    })

    logging.info(f"\n[Requirement ID]: {req_id}")
    logging.info(f"\n[Requirement decomposition answer]: {answer}")
    logging.info("\n----------------------- Requirement decomposition pipeline finished --------------------")

    return answer
