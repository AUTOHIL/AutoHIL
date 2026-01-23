import os
import sys
path1 = os.path.abspath(os.path.dirname(__file__) + '/..')
sys.path.append(path1)
sys.path.append(path1 + '/..')
sys.path.append(path1 + '/../..')
from dotenv import load_dotenv
from typing import List
import logging
import json
import pandas as pd

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

# --- 3. Vector embeddings ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 4. Vector store (ChromaDB) ---
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# --- 5. LLM ---
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# --- utils ---
from utils.excel_parser import ExcelParser


DB_PERSIST_PATH = "./rag_db/req2src_db"

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='./argsFinding/log/req2src.log'
)

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
        raise ValueError("GOOGLE_API_KEY environment variable is not set (for Embeddings)")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    return embeddings

def save_results_to_excel(data_list, filename):
    if not data_list:
        logging.error("data list is none.")
        return
    try:
        df = pd.DataFrame(data_list)
        df.to_excel(filename, index=False, engine="openpyxl")
        logging.debug(f"Successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error while saving results to {filename}")

def format_docs(docs: List[Document]) -> str:
    formatted_strings = []
    for doc in docs:
        if 'function_name' in doc.metadata:
            # Build a format that is easy for the LLM to understand
            func_name = doc.metadata['function_name']
            func_summary = doc.page_content
            formatted_strings.append(f"Function name: {func_name}\nSummary: {func_summary}")
        else:
            # Fallback for other document types (e.g., .txt)
            formatted_strings.append(doc.page_content)
    
    # Separate each function with a delimiter
    return "\n\n---\n\n".join(formatted_strings)

def load_document(file_path: str):
    _, extension = os.path.splitext(file_path)
    logging.info(f"  > Loading: {file_path} (type: {extension})")

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".json":
            docs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f) # data is {"func1": "summary1", ...}
            
            for func_name, func_summary in data.items():
                # We manually create Document objects
                doc = Document(
                    # page_content stores the "summary", which is the searchable content
                    page_content=func_summary, 
                    # metadata stores the "function name", which is the final answer we want
                    metadata={"function_name": func_name, "source": file_path} 
                )
                docs.append(doc)
            return docs
        elif extension in [".txt", ".py", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            logging.warning(f"  > Unsupported file type: {extension}, skipping.({file_path})")
            return []
        
        return loader.load()
    except Exception as e:
        logging.error(f"  > Error while loading file: {file_path}, error: {str(e)}")
        return []

def extract_answer(content):
    start_marker = "```python"
    if start_marker in content:
        start_index = content.find(start_marker) + len(start_marker)

        end_marker = "```"
        end_index = content.rfind(end_marker)

        py_content = content[start_index:end_index ]
    else:
        py_content = content

    return py_content

def create_index(file_paths: List[str]):
    """
    Load, split, embed files, and save the index to disk
    """
    logging.info("  1. Loading documents...")
    all_docs = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_docs.extend(docs)
        else:
            logging.warning(f"  > Warning: skipping non-existent file: {file_path}")

    # Text splitting
    logging.info("  2. Splitting text...")
    splits = all_docs
    logging.info(f"  Number of document chunks (Splits) after splitting: {len(splits)}")

    # Vector embeddings
    logging.info("  3. Generating embedding vectors...")
    embeddings = get_google_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PERSIST_PATH
    )
    logging.info(f"--- Index creation completed and saved to {DB_PERSIST_PATH} ---")


def get_related_source(module_name, req_id, user_query) -> str:
    """
    Load the vector database and execute the RAG query
    """

    if not os.path.exists(DB_PERSIST_PATH):
        logging.error(f"Vector database path does not exist: {DB_PERSIST_PATH}")
        return
    
    # Embed the query in the same way, then compare it with the database
    embeddings = get_google_embeddings()

    vectorstore = Chroma(
        persist_directory=DB_PERSIST_PATH,
        embedding_function=embeddings
    )

    llm = get_gemini_llm()

    # Load the RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 971})
    doc_file = retriever.invoke(user_query)

    template = f"""You are a strict instruction-following [Requirement-Source Code] matching expert. Your task is to analyze a requirement and identify the relevant functions that implement it from the [Given Source Code Library].
    You must strictly adhere to the following rules:
    a. Search Only Within Context: Your response [MUST] come 100% from the functions provided within the <Source Code Functions> tags below.
    b. No Fabrication: It is strictly forbidden to mention, guess, or fabricate any function names that are [NOT explicitly listed] within <Source Code Functions>.
    c. Prioritization Rule: If multiple functions appear relevant, prioritize functions starting with canmgr (provided they exist within <Source Code Functions>).
    d. Module Hint: The module is the primary focus of the current analysis.""" + """
    Please place all identified function names into a Markdown Python code block, formatted as a Python list.
    
    <Source Func>
    {context}
    </Source Func>
    <Requirement>
    {question}
    </Requirement>
    
    [Your Answer]
"""
    prompt = ChatPromptTemplate.from_template(template)
    context = format_docs(doc_file)
    rag_chain = (prompt | llm | StrOutputParser())

    answer = rag_chain.invoke({
        "context": context,
        "question": user_query
    })

    logging.info(f"\n[req_id]: {req_id}")
    logging.info(f"\n[Related function answer]: {answer}\n")

    return extract_answer(answer)

def main():
    # Params Config
    module_name = "CanMgr"
    req_file = "./reqDecomposer/req_aug/requirements.xlsx"
    target_columns = ['req_id','groundtruth', 'original_req', 'augmented_req']
    output_excel_file = "./argsFinding/ImportantFuncs.xlsx"

    parser = ExcelParser(req_file)
    if not parser.load_excel():
        logging.error("Failed to load Excel")
        return
    
    result_data = []
    processed_req_ids = set()

    if os.path.exists(output_excel_file):
        try:
            logging.info(f"Detected existing result file: {output_excel_file}; reading processed requirement IDs...")
            existing_df = pd.read_excel(output_excel_file, engine="openpyxl")
            if 'req_id' in existing_df.columns:
                processed_req_ids = set(existing_df['req_id'].astype(str).tolist())
                # Load existing data into result_data
                result_data = existing_df.to_dict('records') 
                logging.info(f"Successfully loaded {len(processed_req_ids)} processed requirement IDs.")
            else:
                logging.warning(f"Column 'req_id' not found in result file {output_excel_file}; reprocessing all requirements.")
                processed_req_ids = set() # Clear the set to ensure reprocessing
                result_data = []      # Clear the list
        except Exception as e:
            logging.error(f"Failed to read existing result file {output_excel_file}: {e}. Reprocessing all requirements.")
            processed_req_ids = set() # Reprocess on error as well
            result_data = []

    # Start iterating over requirement entries
    target_data = parser.extract_target_data(target_columns)

    if target_data:
        for i, item in enumerate(target_data):
            row_data = item['row_data']
            req_id = row_data.get('req_id')
            req_groundtruth = row_data.get('groundtruth')
            req_org = row_data.get('original_req')
            req_aug = row_data.get('augmented_req')
            
            if req_id in processed_req_ids:
                continue

            # Initialization
            current_result = {
                'req_id': req_id,
                'groundtruth': req_groundtruth,
                'original_req': req_org,
                'augmented_req': req_aug,
                'important_func': 'Error'
            }

            try:
                funcs = get_related_source(module_name, req_id, req_aug)

            except Exception as e:
                logging.error("Exception while finding related functions: {e}")
                funcs = None
            current_result['important_func'] = funcs if funcs else "Error"

            result_data.append(current_result)

            if funcs:
                processed_req_ids.add(req_id)
            
            save_results_to_excel(result_data, output_excel_file)
            logging.info(f"##################### {req_id} processing finished #####################\n")
        
        if result_data:
            result_df = pd.DataFrame(result_data)
            result_df.to_excel(output_excel_file, index=False, engine="openpyxl")
            logging.info(f"All results have been saved to {output_excel_file}")


if __name__ == "__main__":
    main()
