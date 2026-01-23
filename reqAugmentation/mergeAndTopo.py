import os
import re
import sys
import json
from collections import defaultdict, deque
from depClosureUtils.genAbstract import generate_abstract_en

module_name = "CanMgr"
SUMMARY_CACHE_PATH = f"./output/{module_name}/summary/summary_cache_en.json"

class FuncNode:
    def __init__(self, name, file, line_start, line_end, code, macro):
        self.name = name
        self.file = file
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        self.callers = []   # parents
        self.callees = []   # children
        self.is_root = False
        self.is_leaf = False
        self.is_visited = False
        self.macro = macro
    
    def __repr__(self):
        return f"FuncNode({self.name}, {self.file}, {self.line_start}, {self.line_end})"
    

# Parse call edges from DOT files
def parse_dot_edges(dot_dir):
    edge_pattern = re.compile(r'\s*"(.+?)"\s*->\s*"(.+?)";')
    edges = set()

    for filename in os.listdir(dot_dir):
        if filename.endswith('.dot'):
            with open(os.path.join(dot_dir, filename), 'r') as f:
                for line in f:
                    match = edge_pattern.match(line.strip())
                    if match:
                        caller, callee = match.groups()
                        if caller != callee:
                            edges.add((caller, callee))
    return edges

# Build function call forest
def build_call_forest_with_references(edges, function_info_map, dot_path=None):
    graph = {}
    all_involved_funcs = set()

    dot_file = open(dot_path, 'w') if dot_path else None
    if dot_file:
        dot_file.write("digraph CallForest {\n")

    def get_or_create_node(func_name):
        if func_name not in graph:
            info = function_info_map.get(func_name, {
                "file": "unknown_file",
                "lineNumber": -1, 
                "lineNumberEnd": -1,
                "code": "",
                "macro": ""
            })
            node = FuncNode(
                func_name,
                info["file"],
                info["lineNumber"], 
                info["lineNumberEnd"],
                bytes(info["code"], encoding="utf-8").decode("raw_unicode_escape"),
                info["macro"]
            )
            graph[func_name] = node
        
            if dot_file:
                label = f"{func_name}\\n{node.file}:{node.line_start}-{node.line_end}"
                dot_file.write(f'    "{func_name}" [label="{label}"];\n')
        
        return graph[func_name]

    for caller_name, callee_name in edges:
        caller = get_or_create_node(caller_name)
        callee = get_or_create_node(callee_name)
        caller.callees.append(callee)
        callee.callers.append(caller)
        all_involved_funcs.update([caller_name, callee_name])

        if dot_file:
            dot_file.write(f'  "{caller_name}" -> "{callee_name}";\n')
    
    for func_name in function_info_map:
        if func_name not in all_involved_funcs:
            get_or_create_node(func_name)
            all_involved_funcs.add(func_name)
            print(f"Strange function!—— {func_name}")

    if dot_file:
        dot_file.write("}\n")
        dot_file.close()
        print(f"Call graph DOT file write completed: {dot_path}")

    return graph

# -- Update node flags --
# 1. is_root: whether node is a root (no callers)
# 2. is_leaf: whether node is a leaf (no callees)
# 3. is_visited: whether node has been visited (for traversal)
def update_node_flags(graph):
    for node in graph.values():
        node.is_root = len(node.callers) == 0
        node.is_leaf = len(node.callees) == 0

def find_leaf_nodes(graph):
    return [node for node in graph.values() if node.is_leaf]

def find_root_nodes(graph):
    return [node for node in graph.values() if node.is_root]

def find_not_visited_nodes(graph):
    return [node for node in graph.values() if not node.is_visited]

def print_forest_summary(graph):
    print(f"\nTotal function nodes: {len(graph.values())}")
    roots = find_root_nodes(graph)
    print(f"Root node count: {len(roots)}")
    leaves = find_leaf_nodes(graph)
    print(f"Leaf node count: {len(leaves)}")


# ---------- Reverse topological sort ----------
def traverse_bottom_up(graph):
    # record out-degree for each node (how many callees)
    out_degree = {name: len(node.callees) for name, node in graph.items()}
    # start from leaf nodes (no callees)
    queue = deque(find_leaf_nodes(graph))
    levels = defaultdict(list)
    node_level = {}  # node -> level

    for node in queue:
        node_level[node.name] = 0
        levels[0].append(node)

    max_level = 0

    while queue:
        current = queue.popleft()
        current_level = node_level[current.name]

        for parent in current.callers:
            # each parent advances only after all its children are processed
            out_degree[parent.name] -= 1
            if out_degree[parent.name] == 0:
                next_level = current_level + 1
                node_level[parent.name] = next_level
                levels[next_level].append(parent)
                queue.append(parent)
                max_level = max(max_level, next_level)

    return levels, max_level

# Reverse topological sort - result visualization
def write_traverse_paths_to_file(levels, max_level, output_path):
    with open(output_path, "w") as f:
        for level in range(max_level + 1):
            f.write(f"Level {level}:\n")
            for node in levels[level]:
                f.write(f"- {node.name} ({node.file}:{node.line_start}-{node.line_end})\n")
            if level != max_level:
                f.write("↑\n\n")
    print(f"Bottom-up layered traversal results written to: {output_path}")

# ---------- Detect and handle cycles ----------
def detect_cycles(graph):
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node, path):
        if node.name in rec_stack:
            # found a cycle, record the path
            cycle_start = path.index(node.name)
            cycles.append(path[cycle_start:])
            return
        if node.name in visited:
            return
        visited.add(node.name)
        rec_stack.add(node.name)
        for callee in node.callees:
            dfs(callee, path + [callee.name])
        rec_stack.remove(node.name)

    for node in graph.values():
        if node.name not in visited:
            dfs(node, [node.name])
    return cycles


def complete_levels_with_cycles(graph, levels, max_level):
    all_nodes = set(graph.keys())
    traversed_nodes = set()
    for nodes in levels.values():
        traversed_nodes.update(node.name for node in nodes)
    missing_nodes = all_nodes - traversed_nodes
    print(f"missing_nodes: {missing_nodes}")
    if missing_nodes:
        # can create a special level for them, or output them directly
        levels[max_level + 1] = [graph[name] for name in missing_nodes]
    return levels

# -- Summary cache --
def load_summary_cache(path=SUMMARY_CACHE_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# -- Save summary cache --
def save_summary_cache(summary_map, path=SUMMARY_CACHE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_map, f, ensure_ascii=False, indent=2)
    print(f"Summary cache updated and written to: {path}")

# -- Step-wise approach --
# Generate abstracts based on function levels
def generate_abstract_from_levels(levels, max_level, project_path):
    abstract_map = load_summary_cache()
    atomic_api_map = {}
    all_macro_analysis = {}  # Collect macro reference analysis results for all functions


    # --- Count total nodes ---
    total_nodes = sum(len(nodes) for nodes in levels.values())
    processed_nodes = len(abstract_map)
    print(f"\n🚀 Starting abstract generation: {total_nodes} function nodes ({processed_nodes} cached)")

    for level in range(max_level + 1):
        for node in levels[level]:
            if node.name in abstract_map:
                continue

            # --- Process callee abstracts ---
            callee_abstracts = {
                callee.name: abstract_map.get(callee.name, "")
                for callee in node.callees
            }

            # --- Handle macro references ---
            macro_definitions = getattr(node, "macro", "")

            # --- Get source code ---
            current_code = getattr(node, "code", "")

            # --- Generate abstract ---
            abstract = generate_abstract_en(node.name, current_code, callee_abstracts, macro_definitions)
            abstract_map[node.name] = abstract
            processed_nodes += 1
            print_progress_bar(processed_nodes, total_nodes)

        save_summary_cache(abstract_map)
    
    # Handle cyclic nodes
    cyclic_node = [node.name for node in levels[max_level + 1]]
    for node in levels[max_level + 1]:
        if node.name in abstract_map:
            continue
        callee_abstracts = {
            callee.name: abstract_map.get(callee.name, "")
            for callee in node.callees if callee.name not in cyclic_node
        }
        
        # --- Handle macro references ---
        macro_definitions = getattr(node, "macro", "")
        
        current_code = getattr(node, "code", "")
        abstract = generate_abstract_en(node.name, current_code, callee_abstracts, macro_definitions)
        abstract_map[node.name] = abstract
        processed_nodes += 1
        print_progress_bar(processed_nodes, total_nodes)
        
    save_summary_cache(abstract_map)
    
    return abstract_map, atomic_api_map


# -- Write abstract information --
def save_abstract_map(abstract_map, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(abstract_map, f, ensure_ascii=False, indent=2)
    print(f"Abstract info written to: {output_path}")

# -- Progress bar --
def print_progress_bar(current, total, bar_length=30):
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    progress_str = f"Progress: [{bar}] {current}/{total} ({int(percent * 100)}%)"
    print(progress_str, end='\r', flush=True)


if __name__ == "__main__":
        
    # Configure paths
    project_path = f"./projClosure/{module_name}"
    dot_directory = f"./output/{module_name}/cg"
    abstract_directory = f"./output/{module_name}/summary"
    json_path = os.path.join(abstract_directory, "function_info_map.json")
    # Check whether critical paths exist
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    if not os.path.exists(dot_directory):
        raise FileNotFoundError(f"DOT file directory does not exist: {dot_directory}")
    if not os.path.exists(abstract_directory):
        raise FileNotFoundError(f"Summary directory does not exist: {abstract_directory}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"FunctionInfoMap file does not exist: {json_path}")
    bottom_up_path = os.path.join(abstract_directory, "bottomUp.txt")
    abstract_map_path = os.path.join(abstract_directory, "api_abstracts.json")

    # Read JSON
    with open(json_path, "r") as f:
        function_info_map = json.load(f)
    
    if not function_info_map:
        raise ValueError("FunctionInfoMap is empty")

    # Read all call edges
    dot_edges = parse_dot_edges(dot_directory)

    # Build forest graph (reference structure)
    graph = build_call_forest_with_references(
        dot_edges, 
        function_info_map
    )
    update_node_flags(graph)

    # Output summary stats
    print_forest_summary(graph)

    levels, max_level = traverse_bottom_up(graph)
    levels = complete_levels_with_cycles(graph, levels, max_level)
    
    cycles = detect_cycles(graph)
    print(f"Detected cycles: {cycles}")

    abstract_map, atomic_api_map = generate_abstract_from_levels(levels, max_level, project_path)
