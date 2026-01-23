"""End-to-end automation"""
# preparation: 
# 1. HIL API Pool construction
# 2. ReqExtract -- generate modular req corpus
# Test generation:(main)
# 3. Requirement Augmentation
# 4. Requirement Decomposition
# 5. BTEncoder
# 6. BTDecoder(Test generation) + post processing
import os
import sys
import logging
import json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from utils.excel_parser import ExcelParser, save_results_to_excel
from reqAugmentation.reqAugment import req_augment
from reqDecomposer.reqDec_local_rag import query_rag
from reqDecomposer.btEncoder import bt_encoder
from IR2Script.btDecoder import bt_decoder_main
from IR2Script.post_process_rag import post_processing


def main(hil_api_abstract, hil_config_file, hil_api_info, attr_tree_path,
         script_output_dir, script_prefix, 
         module_dict):
    target_columns = ['req_id','groundtruth', 'original_req']
    
    for module_name, paths in module_dict.items():
        req_file = paths["module_req_file"]
        output_excel_file = paths["output_excel_file"]
        module_corpus_file = paths["module_corpus_file"]
        module_config_file = paths["module_config_file"]
        
        # Read Requirement File
        parser = ExcelParser(req_file)
        if not parser.load_excel():
            logging.error("Fail to load Excel")
            continue
        target_data = parser.extract_target_data(target_columns)
        total_reqs = len(target_data)
        logging.info(f"Total {total_reqs} requirements to process for module {module_name}.")
        
        result_data = []
        processed_req_ids = set()
        
        while len(processed_req_ids) < total_reqs:
            if os.path.exists(output_excel_file):
                try:
                    logging.info(f"Detected existing result file: {output_excel_file}, reading processed requirement IDs...")
                    existing_df = pd.read_excel(output_excel_file, engine="openpyxl")
                    if 'req_id' in existing_df.columns:
                        processed_req_ids = set(existing_df['req_id'].astype(str).tolist())
                        result_data = existing_df.to_dict('records')
                        logging.info(f"Loaded {len(processed_req_ids)} processed requirement IDs.")
                except Exception as e:
                    logging.error(f"Error reading existing result file: {e}")
                    
            if target_data:
                for i, item in enumerate(target_data):
                    row_data = item['row_data']
                    req_id = row_data.get('req_id')
                    req_groundtruth = row_data.get('groundtruth')
                    req_org = row_data.get('original_req')
                    
                    if req_id in processed_req_ids:
                        continue
                    
                    current_result = {
                        'req_id': req_id,
                        'groundtruth': req_groundtruth,
                        'original_req': req_org,
                        'augmented_req': "Error",
                        'decomposed_req': "Error",
                        'bt_json': "Error",
                        'directed_code': "Error",
                        'refined_code': "Error"
                    }
            
                    # -------------- ReqAugmentation --------------
                    logging.info(f"-------------------------- {req_id} REQ Augmentation --------------------------")
                    try:
                        augment_knowledge_files = [module_corpus_file, module_config_file]
                        augmented_req = req_augment(req_id, req_org, augment_knowledge_files)
                    except Exception as e:
                        logging.error(f"Requirement Augmentation failed for {req_id}: {e}")
                        augmented_req = None
                    current_result["augmented_req"] = augmented_req if augmented_req else "Error"

                    # -------------- ReqDecomposition --------------
                    if augmented_req:
                        logging.info(f"-------------------------- {req_id} REQ Decomposition --------------------------")
                        try:
                            KNOWLEDGE_FILES = [hil_api_abstract, hil_config_file]
                            decomposed_req = query_rag(req_id, augmented_req, KNOWLEDGE_FILES)
                        except Exception as e:
                            logging.error(f"Requirement Decomposition failed for {req_id}: {e}")
                            decomposed_req = None
                        current_result["decomposed_req"] = decomposed_req

                    # -------------- BTEncoder --------------
                    if augmented_req and decomposed_req:
                        logging.info(f"-------------------------- {req_id} BTEncoder --------------------------")
                        try:
                            bt_json, json_file = bt_encoder(decomposed_req, module_name, req_id)
                        except Exception as e:
                            logging.error(f"BT Encoding failed for {req_id}: {e}")
                            bt_json, json_file = None, None
                        current_result["bt_json"] = bt_json if bt_json else "Error"

                    # -------------- BTDecoder --------------
                    if augmented_req and decomposed_req and bt_json:
                        logging.info(f"-------------------------- {req_id} BTDecoder --------------------------")
                        try:
                            output_py = os.path.join(script_output_dir, module_name, "directed", f"{req_id}.py")
                            generated_code = bt_decoder_main(module_name, req_id, 
                                                            json_file, output_py, hil_api_info, attr_tree_path, 
                                                            script_prefix)
                        except Exception as e:
                            logging.error(f"BT Decoding failed for {req_id}: {e}")
                            generated_code = None
                        current_result["directed_code"] = generated_code if generated_code else "Error"
                    
                    # -------------- Post Processing --------------
                    if augmented_req and decomposed_req and bt_json and generated_code:
                        logging.info(f"-------------------------- {req_id} POST PROCESS --------------------------")
                        try:
                            adjusted_file = os.path.join(script_output_dir, module_name, "adjusted", f"{req_id}.py")
                            refined_code = post_processing(req_id, bt_json, generated_code, adjusted_file)
                        except Exception as e:
                            logging.error(f"Post Processing failed for {req_id}: {e}")
                            refined_code = None
                        current_result["refined_code"] = refined_code if refined_code else "Error"
                    
                    result_data.append(current_result)
                    if augmented_req and decomposed_req and bt_json and generated_code and refined_code:
                        processed_req_ids.add(req_id)
                        
                    save_results_to_excel(result_data, output_excel_file)
                    logging.info(f"-------------------------- {req_id} Finished --------------------------")
                
        if result_data:
            result_df = pd.DataFrame(result_data)
            result_df.to_excel(output_excel_file, index=False, engine="openpyxl")
            logging.info(f"Final results saved to {output_excel_file}")
            
if __name__ == "__main__":
    # Load configuration from config.json (located next to this file).
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            logging.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.warning(f"Configuration file not found: {config_path}. Using defaults.")
    except Exception as e:
        logging.error(f"Failed to load configuration {config_path}: {e}. Using defaults.")

    # HIL API Pool = Abstract + Config + Static Analysis Info + Attribute Tree
    hil_api_abstract = config.get("hil_api_abstract", "./apiPool/abstract/api_abstracts_openai.json")
    hil_config_file = config.get("hil_config_file", "./apiPool/abstract/cfg.py")
    hil_api_info = config.get("hil_api_info", "./apiPool/abstract/function_info_map.json")
    attr_tree_path = config.get("attr_tree_path", "./apiPool/abstract/enhanced_attr_tree.json")

    # Output Script Path
    script_output_dir = config.get("script_output_dir", "./IR2Script/")
    script_prefix = config.get("script_prefix", "")

    # Module Settings - can be provided in config.json under "module_dict"
    config_msg = config.get("config_msg", "./reqAugmentation/req_corpus/MsgDef.py")
    module_dict = config.get("module_dict", {})

    main(hil_api_abstract, hil_config_file, hil_api_info, attr_tree_path,
         script_output_dir, script_prefix,
         module_dict
         )
