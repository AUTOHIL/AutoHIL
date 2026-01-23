import os
import sys
path1 = os.path.abspath(os.path.dirname(__file__) + '/..')
sys.path.append(path1)
sys.path.append(path1 + '/..')
sys.path.append(path1 + '/../..')
import pandas as pd

from reqAugmentation.reqAugment import create_reqAug_index
from reqDecomposer.reqDec_local_rag import create_api_pool_index
from IR2Script.post_process_rag import create_post_process_index


def init(module_name):
    reqAug_knowledge_files = [
        f"./reqAugmentation/req_corpus/{module_name}_corpus.json",
        f"./reqAugmentation/req_corpus/{module_name}Cfg.py",
        "./reqAugmentation/req_corpus/MsgDef.py"
    ]
    create_reqAug_index(reqAug_knowledge_files)
    
    reqDec_knowledge_files = [
        "./apiPool/api_pool/abstract/api_abstracts_openai.json",
        "./apiPool/api_pool/abstract/cfg.py"
    ]
    create_api_pool_index(reqDec_knowledge_files)
    
    postProc_knowledge_files = [
        "./apiPool/api_pool/abstract/api_abstracts_openai.json"
    ]
    create_post_process_index(postProc_knowledge_files)


if __name__ == "__main__":
    module_name = "CanMgr"
    init(module_name)
