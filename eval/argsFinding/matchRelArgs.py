"""
Find variables in source code - intersect with DWARF variables
"""
import os
import sys
path1 = os.path.abspath(os.path.dirname(__file__) + '/..')
sys.path.append(path1)
sys.path.append(path1 + '/..')
sys.path.append(path1 + '/../..')
import logging
import pandas as pd
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.excel_parser import ExcelParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='./argsFinding/log/matchRelArgs.log'
)

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

############## Extract local vars from function source code ################
def read_code_snippet(func_data, func_name):
    try:
        code_snippet = func_data[func_name].get("code", "")
        if not code_snippet:
            logging.warning(f"Function {func_name} did not find the corresponding code snippet.")
        return code_snippet
    except Exception as e:
        logging.error(f"Error while reading the function info file: {e}")
        return ""
    
def get_gemini_llm():
    api_key = os.getenv("GOOGLE_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set (for LLM)")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key
    )

    return llm

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

def step_get_code_vars(code_snippet):
    llm = get_gemini_llm()
    template = """
Find all global variables and static variables in the following code, with the following requirements:
1. For struct variables, output both the top-level variable and the expanded member variables involved in the code, e.g., A.b, A.b.c.
2. For unions, unify the '->' operator to '.'; output both the top-level variable and the expanded member variables involved in the code.
3. For arrays, the index [MUST] be unified as [i]; output both the top-level variable and the expanded member variables involved in the code.
4. Do not output macros or function names; avoid missing variables.
Only output a variable list ([] without a list name), put it in a Python code block; do not output any other explanation.

[Code]
{code_snippet}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "code_snippet": code_snippet
    })
    logging.info(f"{answer}\n")
    response_refined = list(eval(extract_py_code(answer)))

    return response_refined


def batch_get_code_vars(func_info_json_file, output_json_file):
    func_data = {}
    processed_funcs = set()
    
    if os.path.exists(output_json_file):
        logging.info(f"Detected existing progress file: {output_json_file}; loading...")
        try:
            with open(output_json_file, 'r', encoding='utf-8') as out_f:
                func_data = json.load(out_f)

            for func_name, info in func_data.items():
                if "vars" in info:
                    processed_funcs.add(func_name)
            logging.info(f"Loaded {len(processed_funcs)} processed functions")
        except Exception as e:
            logging.error(f"Failed to load progress file")
            func_data = {}
            processed_funcs = set()

    if not func_data:
        logging.info(f"No progress file found; loading new data from {func_info_json_file}...")
        try:
            with open(func_info_json_file, 'r') as f:
                func_data = json.load(f)
            if func_data is None:
                logging.error("Function info data is empty.")
                return False
        except Exception as e:
            logging.error(f"Failed to load input file")
            return False
    

    total_funcs = len(func_data)
    logging.info(f"Total {total_funcs} functions; start processing {total_funcs - len(processed_funcs)} new functions...")

    processed_this_run = len(func_data)
    for func_name, info in func_data.items():
        
        if func_name in processed_funcs:
            continue

        code = info.get("code", "")
        if code:
            info["vars"] = step_get_code_vars(code)
        else:
            info["vars"] = {}
        processed_this_run += 1

        # Save progress immediately
        with open(output_json_file, "w", encoding='utf-8') as f:
            json.dump(func_data, f, indent=2, ensure_ascii=False)

        processed_funcs.add(func_name)

    logging.info(f"Successfully extracted variables and saved data to {output_json_file}")

    return True

############## Intersect and do some fuzzy handling ################
def normalize_var_name(var_str: str) -> str:
    """
    Normalize variable names by replacing all array access indices with a unified '[i]'.

    Examples:
    - "A[i]" -> "A[i]" (unchanged)
    - "A[Msg_ID]" -> "A[i]"
    - "canmgr_MsgMonitor[Msg_ID].u8TimeoutCommCnt" -> "canmgr_MsgMonitor[i].u8TimeoutCommCnt"
    - "A.b[index].c" -> "A.b[i].c"
    """
    # Ensure var_str is a string, in case an empty value is read
    if not isinstance(var_str, str):
        return str(var_str)
    # Use non-greedy match .*? to replace everything inside brackets with [i]
    return re.sub(r'\[.*?\]', '[i]', var_str)

def match_related_args(local_vars: set, dwarf_vars: set):
    try:
        normalized_local_vars = {normalize_var_name(var) for var in local_vars}
        found_vars = dwarf_vars.intersection(normalized_local_vars)
        
        return list(found_vars)
    except Exception as e:
        logging.error(f"Error while matching related args: {e}")
        return []

def main(func_info_json_file):
    # File settings
    related_funcs_excel_file = "./argsFinding/ImportantFuncs.xlsx"          # requirement <-> key function names
    dwarf_vars_csv_file = "./argsFinding/dwarf_file_table_filter.csv"       # DWARF variable parsing table
    target_columns = ['req_id','groundtruth', 'original_req', 'augmented_req', 'key functions']
    output_excel_file = "./argsFinding/ImportantVars.xlsx"

    # Get function summary file info
    func_data = {}
    with open(func_info_json_file, 'r') as f:
        func_data = json.load(f)
    if func_data is None:
        logging.error("Function info data is empty.")
        return

    # Get the DWARF variable list
    dwarf_vars = set(pd.read_csv(dwarf_vars_csv_file)['MEMBER PATH'])
    
    # Traverse requirements, get key function names for each requirement
    parser = ExcelParser(related_funcs_excel_file)
    if not parser.load_excel():
        logging.error("Failed to load related function Excel")
        return
    
    result_data = []
    processed_req_ids = set()

    if os.path.exists(output_excel_file):
        try:
            logging.info(f"Detected existing result file: {output_excel_file}; reading processed requirement IDs...")
            existing_df = pd.read_excel(output_excel_file, engine="openpyxl")
            if 'Requirement ID' in existing_df.columns:
                processed_req_ids = set(existing_df['Requirement ID'].astype(str).tolist())
                logging.info(f"Successfully loaded {len(processed_req_ids)} processed requirement IDs.")
            else:
                logging.warning(f"Column 'Requirement ID' not found in existing result file.")
        except Exception as e:
            logging.error(f"Error while reading existing result file: {e}")
            processed_req_ids = set()
            result_data = []

    logging.info("Start......")
    target_data = parser.extract_target_data(target_columns)
    if target_data:
        for i, item in enumerate(target_data):
            row_data = item['row_data']
            req_id = row_data.get('Requirement ID')
            req_groundtruth = row_data.get('Corresponding groundtruth')
            req_org = row_data.get('Original requirement')
            req_aug = row_data.get('Augmented requirement')
            key_funcs = list(eval(row_data.get('key functions')))

            if req_id in processed_req_ids:
                continue

            if not key_funcs:
                logging.error(f"Requirement ID {req_id} did not find key functions; skipping processing.")
                continue

            current_result = {
                'Requirement ID': req_id,
                'Corresponding groundtruth': req_groundtruth,
                'Original requirement': req_org,
                'Augmented requirement': req_aug,
                'key functions': key_funcs,
                'key variables': "Error"
            }

            related_vars = []
            try:
                for func_name in key_funcs:
                    local_vars = set(func_data[func_name].get("vars", []))
                    if not local_vars:
                        continue
                    related_vars = related_vars + match_related_args(local_vars, dwarf_vars)
            except Exception as e:
                logging.error(f"Error while processing Requirement ID {req_id}: {e}")   
            
            current_result['key variables'] = set(related_vars)
            
            result_data.append(current_result)
            
            if related_vars:
                processed_req_ids.add(req_id)
            
            save_results_to_excel(result_data, output_excel_file)
            logging.info(f"############################## {req_id} processing finished ##############################")
        if result_data:
            result_df = pd.DataFrame(result_data)
            result_df.to_excel(output_excel_file, index=False, engine="openpyxl")
            logging.info(f"All requirements have been processed; results saved to {output_excel_file}")


if __name__ == "__main__":
    
    func_info_json_file = "./argsFinding/abstract/function_info_map.json"   # function name <-> code snippet
    output_info_json_file = "./argsFinding/abstract/function_info_with_vars.json"

    # MODE = "local"  # Extract existing variables in function code
    MODE = "dwarf"  # Intersect with DWARF
    
    if MODE == "local":
        batch_get_code_vars(func_info_json_file, output_info_json_file)
    else:
        main(output_info_json_file)

    logging.info("Finished.")
