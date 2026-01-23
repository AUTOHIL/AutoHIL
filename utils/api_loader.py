import json
import os
from typing import Dict, List, Set, Tuple
config_file_path = "./config.json"

def extract_apis_from_bt_json(bt_json_path: str) -> Set[str]:
    """Extract all API names used in a behavior-tree JSON file"""
    with open(bt_json_path, encoding="utf-8") as f:
        bt_spec = json.load(f)
    
    def traverse_node(node: dict) -> Set[str]:
        apis = set()
        # if the node is a leaf (an atomic API call)
        if "api" in node:
            apis.add(node["api"])
        if "function" in node:
            apis.add(node["function"])
        # recursively traverse child nodes
        if "children" in node:
            for child in node["children"]:
                apis.update(traverse_node(child))
        # handle ForEach node's 'child' field
        if "child" in node:
            apis.update(traverse_node(node["child"]))
        return apis
    
    return traverse_node(bt_spec)

def get_api_imports(api_names: Set[str], function_info_path: str, attr_tree_path: str) -> Tuple[Dict[str, str], List[str]]:
    """Get the corresponding file and class names for APIs from function_info_map.json"""
    with open(function_info_path, encoding="utf-8") as f:
        function_info = json.load(f)
    
    """Load attribute-class mapping from attr_cls_map.json to find attribute paths used when importing APIs"""
    with open(attr_tree_path, encoding="utf-8") as f:
        attr_tree = json.load(f)
    
    with open(config_file_path, 'r') as f:
        project_interface = json.load(f).get("project_interface")
    
    # collect all files that need to be imported and their corresponding APIs
    # file_to_apis: {file: {api: (class, attr_path)}}
    file_to_apis: Dict[str, Dict[str, Tuple[str, str]]] = {}
    not_found_apis: List[str] = []
    
    for api_name in api_names:
        if api_name in function_info:
            file_path = function_info[api_name]["file"]
            if file_path not in file_to_apis:
                file_to_apis[file_path] = {}
            class_name = function_info[api_name]["class"]
            if class_name:
                attr_path = attr_tree[project_interface].get(class_name, "")
                if attr_path:
                    file_to_apis[file_path][api_name] = (class_name, attr_path)
                else:
                    file_to_apis[file_path][api_name] = (class_name, "")
            else:
                file_to_apis[file_path][api_name] = ("", "")
        else:
            not_found_apis.append(api_name)
    
    return file_to_apis, not_found_apis

def generate_import_statements(file_to_apis: Dict[str, Dict[str, Tuple[str, str]]], script_prefix: str = "guardhil_test_script") -> Tuple[str, str]:
    """Generate import statements.

    Returns: (eval_imports, code_imports)
    - eval_imports: import statements used for `execute_imports` execution
    - code_imports: import statements to embed into the generated code
    """
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        project_interface = config.get("project_interface")
        project_interface_path = config.get("project_interface_path")
    
    project_path = project_interface_path.replace(".py", "").replace("/", ".")
    
    # full import statements for execute_imports
    eval_imports = [
        "import sys",
        "import os",
        f"from {script_prefix}.{project_path} import {project_interface}",
        f"{project_interface} = {project_interface}('')"
        ""  # blank line separator
    ]
    
    # import statements for generated code
    code_imports = [
        "import sys",
        "import os",
        "path1 = os.path.abspath(os.path.dirname(__file__) + '/..')",
        "sys.path.append(path1)",
        "sys.path.append(path1 + '/..')",
        "sys.path.append(path1 + '/../..')",
        f"from {script_prefix}.{project_path} import {project_interface}",
        f"{project_interface} = {project_interface}('')"
        ""  # blank line separator
    ]
    
    for item in file_to_apis.items():
        file_path = item[0]
        apis_dict = item[1]

        # convert file path to Python module path
        module_path = file_path.replace(".py", "").replace("/", ".")
    
        for api, (class_name, attr_path) in apis_dict.items():
            if project_interface:
                if class_name == project_interface and attr_path == "":
                    eval_imports.append(f"{api} = {project_interface}.{api}")
                    code_imports.append(f"{api} = {project_interface}.{api}")
                elif attr_path:
                    eval_imports.append(f"{api} = {project_interface}.{attr_path}.{api}")
                    code_imports.append(f"{api} = {project_interface}.{attr_path}.{api}")
                elif class_name:
                    eval_imports.append(f"from {script_prefix}.{module_path} import {class_name}")
                    code_imports.append(f"from {script_prefix}.{module_path} import {class_name}")
                    eval_imports.append(f"{class_name} = {class_name}()")
                    code_imports.append(f"{class_name} = {class_name}()")
                    eval_imports.append(f"{api} = {class_name}.{api}")
                    code_imports.append(f"{api} = {class_name}.{api}")
                    pass
            
            else:
                # fallback: derive class name from file name
                class_name = os.path.splitext(os.path.basename(file_path))[0]

                # generate full import statements for execute_imports
                eval_imports.append(f"from {script_prefix}.{module_path} import {class_name}")
                eval_imports.append(f"{class_name} = {class_name}({class_name})")
                # add class import statements for the generated code
                code_imports.append(f"from {script_prefix}.{module_path} import {class_name}")
                code_imports.append(f"{class_name} = {class_name}({class_name})")
                eval_imports.append(f"{api} = {class_name}.{api}")
                code_imports.append(f"{api} = {class_name}.{api}")
    
    return "\n".join(eval_imports), "\n".join(code_imports)

def generate_api_map(api_names: Set[str], file_to_apis: Dict[str, Dict[str, Tuple[str, str]]]) -> str:
    """Generate code for the `api_map` dictionary"""
    # create reverse mapping: api -> file
    api_to_file = {}
    for file_path, apis_dict in file_to_apis.items():
        for api in apis_dict.keys():
            api_to_file[api] = apis_dict[api][1]
    
    # generate api_map entries
    api_map_items = []
    for api in api_names:
        if api in api_to_file:
            attr_path = api_to_file[api]
            api_map_items.append(f"    \"{api}\": {attr_path}.{api}")
    
    return "{\n" + ",\n".join(api_map_items) + "\n}"

def update_bt2code_imports(bt_json_path: str, function_info_path: str, attr_tree_path: str, script_prefix: str = "guardhil_test_script") -> Tuple[str, str, List[str]]:
    """Update BT2Code.py import statements and API mappings"""
    # extract API names from the BT JSON
    api_names = extract_apis_from_bt_json(bt_json_path)
    
    # obtain import information for APIs
    file_to_apis, not_found_apis = get_api_imports(api_names, function_info_path, attr_tree_path)
    
    # generate import statements
    eval_imports, code_imports = generate_import_statements(file_to_apis, script_prefix)
    
    return eval_imports, code_imports, not_found_apis
