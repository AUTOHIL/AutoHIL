import ast
import os

def get_custom_prefixes(project_path):
    """Automatically get first-level subdirectory names under project path as custom package prefixes"""
    prefixes = []
    for name in os.listdir(project_path):
        full_path = os.path.join(project_path, name)
        if os.path.isdir(full_path):
            # Only consider it a package if the directory contains .py files or __init__.py
            if any(f.endswith('.py') for f in os.listdir(full_path)) or "__init__.py" in os.listdir(full_path):
                prefixes.append(name)
    return prefixes

def is_custom_module(module_name, custom_prefixes):
    return any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in custom_prefixes)

def get_imports_from_file(file_path, custom_prefixes):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_custom_module(alias.name, custom_prefixes):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and is_custom_module(node.module, custom_prefixes):
                imports.add(node.module)

    return list(imports)

def module_to_path(module_name, project_path):
    """Convert module name to file path"""
    rel_path = module_name.replace('.', '/')
    py_path = os.path.join(project_path, rel_path + ".py")
    init_path = os.path.join(project_path, rel_path, "__init__.py")
    if os.path.isfile(py_path):
        return py_path
    elif os.path.isfile(init_path):
        return init_path
    else:
        return None

def collect_all_imports(file_path, project_path, custom_prefixes, visited=None):
    if visited is None:
        visited = set()
    result = set()
    result.add(file_path)
    imports = get_imports_from_file(file_path, custom_prefixes)
    for module in imports:
        module_path = module_to_path(module, project_path)
        if module_path and module_path not in visited:
            visited.add(module_path)
            result.add(module_path)
            # Recursively analyze imported modules
            result.update(collect_all_imports(module_path, project_path, custom_prefixes, visited))
    return result

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Configuration file {config_path} not found. Please ensure the file exists.")
    
    # Project root path
    project_path = config.get("project_prefix", "<your_hil_project_dir_prefix>")
    # The .py file to start analysis from
    file_path = config.get("project_interface_path", "<your_hil_project_entry_path>")

    custom_prefixes = get_custom_prefixes(project_path)
    all_imports = collect_all_imports(file_path, project_path, custom_prefixes)
    # Convert to relative path (remove project_path prefix)
    all_imports = [os.path.relpath(p, project_path) for p in all_imports]
    import json
    with open('./tarFiles.json', 'w', encoding='utf-8') as f:
        json.dump(all_imports, f, ensure_ascii=False, indent=4)
