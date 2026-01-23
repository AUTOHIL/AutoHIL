#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced DOT file function call graph merge script
Supports automatically matching DOT files by function name for merging
"""

import re
import os
import glob
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional

class AdvancedDotCallGraphMerger:
    """Advanced DOT file function call graph merger"""
    
    def __init__(self):
        self.function_to_file_map = {}  # mapping from function name to file path
        self.processed_functions = set()  # processed functions
        
    def parse_dot_file(self, dot_file_path: str) -> Dict:
        """Parse a DOT file and extract node and edge information"""
        try:
            with open(dot_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(dot_file_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # Remove comments
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse result
        result = {
            'nodes': set(),
            'edges': set(),
            'node_attributes': {},
            'graph_name': '',
            'graph_type': 'digraph',
            'file_path': dot_file_path
        }
        
        # Extract graph name and type
        graph_match = re.search(r'(di)?graph\s+(\w+|\".+?\")\s*\{', content)
        if graph_match:
            result['graph_type'] = 'digraph' if graph_match.group(1) else 'graph'
            result['graph_name'] = graph_match.group(2).strip('"')
        
        # Extract node definitions (with attributes)
        node_pattern = r'(\w+|\"[^\"]+\")\s*\[([^\]]+)\]'
        for match in re.finditer(node_pattern, content):
            node = match.group(1).strip('"')
            attributes = match.group(2)
            result['nodes'].add(node)
            result['node_attributes'][node] = attributes
        
        # Extract edges (supports directed and undirected)
        edge_pattern = r'(\w+|\"[^\"]+\")\s*(->|--)\s*(\w+|\"[^\"]+\")'
        for match in re.finditer(edge_pattern, content):
            from_node = match.group(1).strip('"')
            to_node = match.group(3).strip('"')
            result['nodes'].add(from_node)
            result['nodes'].add(to_node)
            result['edges'].add((from_node, to_node))
        
        return result
    
    def build_function_file_map(self, search_dirs: List[str]):
        """Build the mapping from function name to DOT file"""
        self.function_to_file_map = {}
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                print(f"Warning: directory {search_dir} does not exist")
                continue
                
            # Find all DOT files
            dot_files = []
            for pattern in ['*.dot', '**/*.dot']:
                dot_files.extend(glob.glob(os.path.join(search_dir, pattern), recursive=True))
            
            for dot_file in dot_files:
                try:
                    graph_data = self.parse_dot_file(dot_file)
                    # Infer the main function name from the filename
                    filename = Path(dot_file).stem
                    
                    # Clean the filename (remove possible suffixes)
                    clean_filename = re.sub(r'[_-]?(callgraph|call|graph)$', '', filename)
                    
                    # Check whether the graph contains a root node with the same name
                    root_nodes = self.find_root_nodes(graph_data)
                    
                    # Prefer a root node that matches the filename
                    matched_function = None
                    for root in root_nodes:
                        if self.function_name_match(clean_filename, root):
                            matched_function = root
                            break
                    
                    # If no matching root node, use the cleaned filename
                    if not matched_function:
                        matched_function = clean_filename
                    
                    self.function_to_file_map[matched_function] = dot_file
                    # print(f"Mapping: {matched_function} -> {dot_file}")
                    
                except Exception as e:
                    print(f"Error while parsing file {dot_file}: {e}")
    
    def function_name_match(self, name1: str, name2: str) -> bool:
        """Check whether two function names match (supports fuzzy matching)"""
        # Clean function names (remove parentheses, underscores, etc.)
        clean1 = re.sub(r'[^\w]', '', name1.lower())
        clean2 = re.sub(r'[^\w]', '', name2.lower())
        
        return clean1 == clean2 or clean1 in clean2 or clean2 in clean1
    
    def find_leaf_nodes(self, graph_data: Dict) -> Set[str]:
        """Find leaf nodes in the graph (nodes with out-degree 0)"""
        all_nodes = graph_data['nodes']
        has_outgoing = set()
        
        for from_node, to_node in graph_data['edges']:
            has_outgoing.add(from_node)
        
        return all_nodes - has_outgoing
    
    def find_root_nodes(self, graph_data: Dict) -> Set[str]:
        """Find root nodes in the graph (nodes with in-degree 0)"""
        all_nodes = graph_data['nodes']
        has_incoming = set()
        
        for from_node, to_node in graph_data['edges']:
            has_incoming.add(to_node)
        
        return all_nodes - has_incoming
    
    def find_extension_files(self, leaf_nodes: Set[str]) -> List[str]:
        """Find extensible DOT files based on leaf nodes"""
        extension_files = []
        
        for leaf_node in leaf_nodes:
            # Direct match
            if leaf_node in self.function_to_file_map:
                extension_files.append(self.function_to_file_map[leaf_node])
                continue
            
            # Fuzzy match
            for func_name, file_path in self.function_to_file_map.items():
                if self.function_name_match(leaf_node, func_name):
                    extension_files.append(file_path)
                    break
        
        return extension_files
    
    def merge_recursively(self, main_graph: Dict, max_depth: int = 5) -> Dict:
        """Recursively merge graphs, supporting multi-level expansion"""
        merged = {
            'nodes': set(main_graph['nodes']),
            'edges': set(main_graph['edges']),
            'node_attributes': dict(main_graph['node_attributes']),
            'graph_name': main_graph['graph_name'] + '_merged',
            'graph_type': main_graph['graph_type']
        }
        
        current_depth = 0
        queue = deque([merged])
        
        while queue and current_depth < max_depth:
            current_graph = queue.popleft()
            leaf_nodes = self.find_leaf_nodes(current_graph)
            
            # Filter out processed functions
            unprocessed_leaves = leaf_nodes - self.processed_functions
            
            if not unprocessed_leaves:
                break
            
            extension_files = self.find_extension_files(unprocessed_leaves)
            
            if not extension_files:
                break
            
            # print(f"Depth {current_depth + 1} expansion: found {len(extension_files)} extension files")
            
            merged_any = False
            for ext_file in extension_files:
                try:
                    ext_graph = self.parse_dot_file(ext_file)
                    ext_roots = self.find_root_nodes(ext_graph)
                    
                    # Find connection points
                    connecting_nodes = unprocessed_leaves & ext_roots
                    
                    if connecting_nodes:
                        # print(f"  Connection points: {connecting_nodes} (from {ext_file})")
                        
                        # Merge graphs
                        merged['nodes'].update(ext_graph['nodes'])
                        merged['edges'].update(ext_graph['edges'])
                        merged['node_attributes'].update(ext_graph['node_attributes'])
                        
                        # Mark processed functions
                        self.processed_functions.update(connecting_nodes)
                        
                        merged_any = True
                        
                except Exception as e:
                    print(f"  Error while processing {ext_file}: {e}")
            
            if merged_any:
                queue.append(merged)
                current_depth += 1
            else:
                break
        
        print(f"Recursive expansion completed with {current_depth} levels")
        return merged
    
    def generate_dot_content(self, graph_data: Dict) -> str:
        """Generate DOT file content"""
        lines = []
        graph_type = graph_data['graph_type']
        graph_name = graph_data['graph_name']
        edge_symbol = '->' if graph_type == 'digraph' else '--'
        
        lines.append(f'{graph_type} {graph_name} {{')
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, style=filled, fillcolor=lightblue];')
        lines.append('    edge [color=black, fontsize=10];')
        lines.append('')
        
        # Add edges
        for from_node, to_node in sorted(graph_data['edges']):
            lines.append(f'    "{from_node}" {edge_symbol} "{to_node}";')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def process(self, main_dot_file: str, search_dirs: List[str], output_file: str, max_depth: int = 5):
        """Process the main function and merge all related call graphs"""
        print(f"Processing main DOT file: {main_dot_file}")
        
        # Parse the main graph
        if not os.path.exists(main_dot_file):
            print(f"Error: main DOT file {main_dot_file} does not exist")
            return
        
        main_graph = self.parse_dot_file(main_dot_file)
        print(f"Main graph contains {len(main_graph['nodes'])} nodes and {len(main_graph['edges'])} edges")
        
        # Build mapping from function to file
        self.build_function_file_map(search_dirs)
        print(f"Found {len(self.function_to_file_map)} function-to-file mappings")
        
        # Recursively merge
        merged_graph = self.merge_recursively(main_graph, max_depth)
        
        # Generate output file
        output_content = self.generate_dot_content(merged_graph)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"Merged graph contains {len(merged_graph['nodes'])} nodes and {len(merged_graph['edges'])} edges")
        print(f"Output file: {output_file}")

def main(main_dir: str, main_dot: List[str], search_dirs: List[str], output_dir: str):
    """Main function - configure file paths and parameters here"""
    max_depth = 100                                       # maximum recursion depth (prevent infinite expansion)
    for dot in main_dot:
        main_dot_file = f"{main_dir}/{dot}"
        output_file = f"{output_dir}/{dot}"
        print(f"Output file: {output_file}")
    
        merger = AdvancedDotCallGraphMerger()
        merger.process(main_dot_file, search_dirs, output_file, max_depth)

if __name__ == "__main__":
    module_name = "CanMgr"
    main_dir = f"./output/{module_name}/cg/1"
    main_dot = [f"{module_name}_Initialization.dot",
                f"{module_name}_MainFunction.dot"]              # main call graph DOT files (relative to script location)
    search_dirs = [f"./output/{module_name}/cg/2"]              # list of search directories (can be multiple)
    output_dir = f"./output/{module_name}/cg"
    main(main_dir, main_dot, search_dirs, output_dir)
