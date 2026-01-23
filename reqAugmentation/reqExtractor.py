import os
import glob
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

load_dotenv()
# dynamically get the log directory under the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'log')
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
log_file = os.path.join(log_dir, 'gemini_reqExtractor.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


# Utils
def load_document(file_path: str) -> list[Document]:
    """(helper) Dynamically choose and load a single file based on its extension."""
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
        elif extension in [".txt", ".py", ".md", ".dot"]:
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
    """Merge the retrieved Document object list into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_answer(content: str) -> str:
    start_marker = "```json"
    if start_marker in content:
        start_index = content.find(start_marker) + len(start_marker)

        end_marker = "```"
        end_index = content.rfind(end_marker)

        json_content = content[start_index:end_index]
    else:
        json_content = content

    return json_content

def create_index(file_paths: List[str], DB_PERSIST_PATH: str):
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
    logging.info(f"\n  > Document loading complete, total {len(all_docs)} documents.")

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

    # --- 3. Vector embeddings
    logging.info("  3. Initializing embedding model (Google Gemini API)...")
    embeddings = get_google_embeddings()

    logging.info("  4. Creating ChromaDB vector index (in-memory mode)...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=DB_PERSIST_PATH
    )
    
    logging.info(f"--- Index creation completed and saved to {DB_PERSIST_PATH} ---")


# ===================================================================
# RAG flow main function
# ===================================================================

def query_rag(module_name, user_query, knowledge_files, DB_PERSIST_PATH):
    """
    Load the vector DB from disk and perform a RAG query.
    """
    logging.info("\n[--------------------------- Requirement corpus ---------------------------]")

    # 1. Check whether the database exists
    if not os.path.exists(DB_PERSIST_PATH):
        logging.error(f"Error: vector database folder does not exist; please create the index first.")
        return
    
    # 2. Initialize embeddings
    embeddings = get_google_embeddings()

    # 3. Load the database
    vectorstore = Chroma(
        persist_directory=DB_PERSIST_PATH,
        embedding_function=embeddings
    )

    # 4. Initialize LLM
    llm = get_gemini_llm()
    
    # 5. RAG chain
    retriever1 = vectorstore.as_retriever(
        search_kwargs={
            "k": 200,
            "filter": {"source": knowledge_files[0]}
        }
    )
    retriever2 = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {"source": knowledge_files[1]}
        }
    )
    retriever3 = vectorstore.as_retriever(  
        search_kwargs={
            "k": 10,
            "filter": {"source": knowledge_files[2]}
        }
    )
    doc_file1 = retriever1.invoke(user_query)
    doc_file2 = retriever2.invoke(user_query)
    doc_file3 = retriever3.invoke(user_query)
    all_docs = doc_file1 + doc_file2 + doc_file3
    logging.info(f"Abstract File Blocks Retrieved: {len(doc_file1)}")
    logging.info(f"CG DOT File Blocks Retrieved {len(doc_file2) + len(doc_file3)} document chunks")

    template = """
Role: You are a professional expert in Automotive ECU Test Requirement Analysis and Requirement Localization.
Input: I will provide you with the key function call chain of the CANC module source code (.dot file) and the function summary for each function in the chain (.json file).
Task: Please think step-by-step and generate a requirement corpus in JSON format.
1. Analyze the Workflow: Analyze the function call chain of the CANC module to clarify the overall workflow of the module. Identify core entry functions, key processing nodes, branching points, and aggregation points.
2. Extract Requirements: Based on the functional summaries, control flow descriptions, and potential test points provided in the function summaries, extract the module's requirement points, test points, and function points. Each test point must be an independent function, independent operation, or result verification.
3. Ensure Coverage: Focus heavily on the functional points related to the module itself, ensuring the coverage of the entire module is as complete as possible.
JSON Field Format Requirements:
{{
    "req_id": "Auto-increment ID",
    "traceability": "Derived from which function summary (provide the function name)",
    "requirement_point": "Describe the functional requirement/test point using natural language",
    "controlled_reqpoint": "Optional. Describe the requirement/test point using controlled natural language: 'WHEN <condition> THEN <action> WITHIN <time>', or 'WHEN <action> THEN <verification>'. If specific values for <condition>, <action>, or <time> are clearly specified, fill them in; otherwise, do not fabricate values.",
    "trigger": {{ // Trigger condition block, fill as needed"object": "Key variable/Trigger variable",
        "operator": "Operator or text description of condition change",
        "threshold": "Judgment threshold or symbolic constant",
        "unit": "Physical unit"
     }},
     "expected": {{ // Expected result block"action": "Describe the expected behavior in natural language",
         "target": "Optional, expected result value",
         "duration": "Duration"
     }},
     "completeness": "A value between 0-1 indicating whether the information is complete",
     "confidence": "A value between 0-1 indicating the Large Model's confidence in this extraction",
     "related_ids": ["List of related requirement IDs (for cross-sentence scenarios)"]
}}
Output Constraint: Output only the final complete JSON code block of the requirement corpus.

[Call Graph]
{context_cg}

[Function Abstract]
{context_abstract}

[Requirement Entry]
{question}

Your answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    context_abstract = format_docs(doc_file1)
    context_cg = format_docs(doc_file2) + format_docs(doc_file3)

    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({
        "context_cg": context_cg,
        "context_abstract": context_abstract,
        "question": user_query
    })

    logging.info(f"\n[Module]: {module_name}")
    logging.info(f"\n[Requirement corpus export]: {answer}")
    logging.info("\n----------------------- Requirement decomposition pipeline finished --------------------")
    
    return answer

if __name__ == "__main__":
    # Module Name
    module_name = "CanMgr"
    DB_PERSIST_PATH = f"./rag_db/reqExtractor/{module_name}_db"
    
    # RAG Files
    dot_dir = f"./output/{module_name}/cg"
    dot_files = glob.glob(os.path.join(dot_dir, '*.dot'))
    abstract_file = f"./output/{module_name}/summary/summary_cache_en.json"
    KNOWLEDGE_FILES = [abstract_file] + dot_files
    
    # Output Path
    output_file = f"./output/{module_name}/req_corpus/{module_name}_corpus.json"
    
    # MODE = "index"
    MODE = "query"
    
    try:
        if MODE == "index":
            create_index(KNOWLEDGE_FILES, DB_PERSIST_PATH)
        else:
            user_query = "Please analyze all possible requirement test points and requirement statements in this complete module, and output a complete requirement corpus stored as a JSON code block according to the required format. Only output the JSON code block storing the complete requirement corpus."
            answer = query_rag(module_name, user_query, KNOWLEDGE_FILES, DB_PERSIST_PATH)
            
            corpus_content = format_answer(answer)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corpus_content)
            logging.info(f"\n--- Requirement corpus has been saved to: {output_file} ---\n")
            
    except Exception as e:
        logging.error(f"Something error while rag process: {e}")
