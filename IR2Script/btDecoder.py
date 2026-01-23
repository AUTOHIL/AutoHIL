import json
import sys
import os
import importlib
path1 = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(path1)
sys.path.append(path1 + '/..')
sys.path.append(path1 + '/../..')
from typing import Any, Dict, List, Callable
# from reqDecomposer.behaviorTree import BTNode
# from reqDecomposer.api_loader import update_bt2code_imports
from utils.behaviorTree import BTNode
from utils.api_loader import update_bt2code_imports


# ------------------------------------------------------------
# Entry: build + export source code
# ------------------------------------------------------------
def load_bt_from_json(path: str) -> BTNode:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return BTNode.make(spec)


def export_tick_function(root: BTNode, func_name: str = "run_tests") -> str:
    body = root.to_code(indent=1)
    return f"def {func_name}():\n{body}"


def execute_imports(eval_imports: str) -> Dict[str, Any]:
    """
    Manually execute a code block line by line that contains import and assignment statements.
    Return the namespace generated after execution.
    """
    namespace = {}
    
    # Pre-inject sys and os into the namespace so statements like sys.path.append can work
    namespace['sys'] = sys
    namespace['os'] = os
    
    lines = eval_imports.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        try:
            if line.startswith('import '):
                # e.g., "import sys"
                parts = line.split()
                module_name = parts[1]
                namespace[module_name] = importlib.import_module(module_name)
            elif line.startswith('from '):
                # e.g., "from module.sub import ClassName" or "from module import func"
                parts = line.split()
                module_path = parts[1]
                names_to_import = [name.strip() for name in ' '.join(parts[3:]).split(',')]
                
                module = importlib.import_module(module_path)
                for name in names_to_import:
                    namespace[name] = getattr(module, name)
            elif '=' in line:
                # e.g., "my_inst = MyClass()"
                var_name, expression = [part.strip() for part in line.split('=', 1)]
                # Use eval to execute the RHS expression in the existing namespace
                namespace[var_name] = eval(expression, globals(), namespace)
            elif line.startswith('sys.path.append'):
                 # e.g., "sys.path.append(...)"
                 eval(line, globals(), namespace)

        except Exception as e:
            print(f"Error while manually parsing imports: '{line}'")
            print(f"Error type: {type(e).__name__}, Error message: {e}")

    return namespace


def bt_decoder_main(module_name, requirement_num,
                    json_file, output_py, function_info_path, attr_tree_path, 
                    script_prefix):
    # ----------- Auto-generate API imports and mapping ------------
    try:
        eval_imports, code_imports, not_found_apis = update_bt2code_imports(
            json_file, 
            function_info_path,
            attr_tree_path,
            script_prefix
        )
        
        if not_found_apis:
            print("Warning: The following APIs were not found in function_info_map.json:")
            for api in not_found_apis:
                print(f"  - {api}")
        
        # ----------- Generate full Python code ------------
        generated_code = f"""{code_imports}

{export_tick_function(load_bt_from_json(json_file))}

if __name__ == "__main__":
    run_tests()
"""
        
        # ----------- Write to file ------------
        with open(output_py, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        
        print("Code generated successfully! Path: ", output_py)
        return generated_code
    except Exception as e:
        print(e)
        return None
