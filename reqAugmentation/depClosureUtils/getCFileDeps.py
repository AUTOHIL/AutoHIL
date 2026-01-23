import os
import re
import json
import shutil
from typing import List, Set, Optional, Dict

# ---------------------------------------------------------------------------
# Regex: capture all common #include forms, e.g.
#   #include <Std_Types.h>
#   #include "path/to/Header.hpp"
#   #  include<Another.h>
# ---------------------------------------------------------------------------
INCLUDE_PATTERN = re.compile(r"#\s*include\s*[<\"]\s*([^<>\"\s]+(?:/[^<>\"\s]+)*)\s*[>\"]")

# Global cached index {filename: [abs_path1, abs_path2, ...]}
_HEADER_INDEX: Optional[Dict[str, List[str]]] = None
_SOURCE_INDEX: Optional[Dict[str, List[str]]] = None
SOURCE_EXTS = (".c", ".cpp", ".cxx")


def build_header_index(project_root: str) -> Dict[str, List[str]]:
    """Scan all .h/.hpp files under the project root and build an index by filename."""
    global _HEADER_INDEX
    if _HEADER_INDEX is not None:
        return _HEADER_INDEX

    index: Dict[str, List[str]] = {}
    for root, _dirs, files in os.walk(project_root):
        for fname in files:
            if fname.endswith((".h", ".hpp")):
                full = os.path.abspath(os.path.join(root, fname))
                index.setdefault(fname, []).append(full)
    _HEADER_INDEX = index
    return index


def build_source_index(project_root: str) -> Dict[str, List[str]]:
    """Scan all .c/.cpp/.cxx files under the project root and build an index by filename."""
    global _SOURCE_INDEX
    if _SOURCE_INDEX is not None:
        return _SOURCE_INDEX

    index: Dict[str, List[str]] = {}
    for root, _dirs, files in os.walk(project_root):
        for fname in files:
            if fname.endswith(SOURCE_EXTS):
                full = os.path.abspath(os.path.join(root, fname))
                index.setdefault(fname, []).append(full)
    _SOURCE_INDEX = index
    return index


def find_best_match(candidate_paths: List[str], reference_path: str) -> Optional[str]:
    """Find the candidate path with the highest overlap with the reference path (header path).
    The path similarity is judged based on the longest common path.
    """
    if not candidate_paths:
        return None
    if len(candidate_paths) == 1:
        return candidate_paths[0]

    best_match = None
    max_common_len = -1

    # Use the reference path (header path) as the baseline
    abs_reference_dir = os.path.abspath(os.path.dirname(reference_path))

    for path in candidate_paths:
        abs_candidate_dir = os.path.abspath(os.path.dirname(path))
        try:
            # commonpath may raise ValueError on different drives (Windows)
            common = os.path.commonpath([abs_reference_dir, abs_candidate_dir])
            common_len = len(common)
            if common_len > max_common_len:
                max_common_len = common_len
                best_match = path
        except ValueError:
            continue # skip incomparable paths (e.g., different drives)

    # If no best match is found (e.g., all paths raised ValueError), fall back to the first one
    if best_match is None:
        return candidate_paths[0]

    return best_match


def find_source_for_header(header_path: str, project_root: str, entry_file: str) -> List[str]:
    """Try to find corresponding source files (.c/.cpp/.cxx) for a header path.

    Search order:
    1. Search for same-named source file in the same directory as the header
    2. Search for same-named source file in the parent's Source directory
    3. Search for same-named source file using the project-wide index
    4. Fallback strategy:
       - If the header is under a "Public" directory, return all .c files under the parent's "Source" directory
       - Otherwise, return all .c files under the header's directory
    """
    base_name = os.path.splitext(os.path.basename(header_path))[0]
    header_dir = os.path.dirname(header_path)

    # 1) Search for same-named source file in the same directory as the header
    for ext in SOURCE_EXTS:
        candidate = os.path.join(header_dir, base_name + ext)
        if os.path.isfile(candidate):
            return [os.path.abspath(candidate)]

    # 2) Search for same-named source file in the parent's Source directory
    parent_dir = os.path.dirname(header_dir)
    source_dir = os.path.join(parent_dir, "Source")
    for ext in SOURCE_EXTS:
        candidate = os.path.join(source_dir, base_name + ext)
        if os.path.isfile(candidate):
            return [os.path.abspath(candidate)]

    # 3) If still not found, search for same-named source file using the project-wide index
    index = build_source_index(project_root)
    for ext in SOURCE_EXTS:
        fname = base_name + ext
        paths = index.get(fname)
        if paths:
            # Take the first two path components of header_path relative to project_root as the matching prefix
            header_rel_path = os.path.relpath(header_path, project_root)
            path_components = header_rel_path.split(os.sep)
            if len(path_components) >= 2:
                prefix_rel_path = os.path.join(path_components[0], path_components[1])
            else:
                prefix_rel_path = path_components[0] if path_components else ""
            
            # Filter files whose relative path prefix matches header_path
            matching_files = []
            for path in paths:
                candidate_rel_path = os.path.relpath(path, project_root)
                if candidate_rel_path.startswith(prefix_rel_path):
                    matching_files.append(path)
            
            if matching_files:
                return matching_files

    # 4) Fallback strategy: collect all .c files under the directory
    search_dir = header_dir
    if os.path.basename(header_dir) == "Public":
        parent_dir = os.path.dirname(header_dir)
        search_dir = os.path.join(parent_dir, "Source")

    c_files = []
    if not os.path.isdir(search_dir):
        return c_files

    try:
        # Collect all .c files under the directory
        for fname in os.listdir(search_dir):
            if fname.endswith('.c'):
                c_files.append(os.path.abspath(os.path.join(search_dir, fname)))
    except OSError:
        pass
    
    return c_files


def extract_includes_from_file(file_path: str) -> List[str]:
    """Parse the file and return a list of all included header strings (dedup while preserving order)."""
    includes: List[str] = []
    seen: Set[str] = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = INCLUDE_PATTERN.search(line)
                if m:
                    header = m.group(1).strip()
                    if header not in seen:
                        includes.append(header)
                        seen.add(header)
    except FileNotFoundError:
        pass
    return includes


def find_header_path(include: str, current_dir: str, project_root: str, entry_file: str) -> Optional[str]:
    """Infer the absolute header path from an include string. Search order:
    1. Current directory relative path
    2. Project root relative path
    3. Global index (search the entire project by filename)"""
    # 1) Current directory
    candidate = os.path.join(current_dir, include)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # 2) Project root
    candidate = os.path.join(project_root, include)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # 3) Global index
    index = build_header_index(project_root)
    paths = index.get(os.path.basename(include))
    if paths:
        # Take the first two path components of entry_file relative to project_root as the matching prefix
        entry_rel_path = os.path.relpath(entry_file, project_root)
        path_components = entry_rel_path.split(os.sep)
        if len(path_components) >= 2:
            prefix_rel_path = os.path.join(path_components[0], path_components[1])
        else:
            prefix_rel_path = path_components[0] if path_components else ""
        
        # Filter files whose relative path prefix matches entry_file
        for path in paths:
            candidate_rel_path = os.path.relpath(path, project_root)
            if candidate_rel_path.startswith(prefix_rel_path):
                return path
        
        # If no match is found, return the first candidate file as a fallback
        return paths[0]
    return None


def collect_headers(file_path: str, project_root: str, entry_file: str, visited: Optional[Set[str]] = None) -> Set[str]:
    """Recursively collect all header absolute paths directly or indirectly depended on by file_path."""
    if visited is None:
        visited = set()

    result: Set[str] = set()

    if not os.path.isfile(file_path):
        return result

    # Add the current file itself to the result
    result.add(os.path.abspath(file_path))

    current_dir = os.path.dirname(file_path)
    for header in extract_includes_from_file(file_path):
        header_path = find_header_path(header, current_dir, project_root, entry_file)
        if header_path and header_path not in visited:

            # Exclude files outside the project root (e.g., system libraries)
            if not header_path.startswith(project_root):
                continue
            
            visited.add(header_path)
            result.add(header_path)
            result.update(collect_headers(header_path, project_root, entry_file, visited))

            # Find corresponding source files and recurse
            source_paths = find_source_for_header(header_path, project_root, entry_file)
            for source_path in source_paths:
                if source_path not in visited:
                    visited.add(source_path)
                    result.add(source_path)
                    result.update(collect_headers(source_path, project_root, entry_file, visited))

    return result


# Match all forms of include, including <> and ""
_ALL_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^<>\"]+)[>\"]")


def save_headers_to_json(headers: Set[str], project_root: str, json_path: str):
    """Save relative paths to json."""
    rel_paths = []
    for p in sorted(headers):
        if p.startswith("UNIFIED_HEADER:"):
            # Unified header special marker: save directly
            rel_paths.append(p)
        else:
            # Normal files: save relative paths
            rel_paths.append(os.path.relpath(p, project_root))
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rel_paths, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(rel_paths)} files to {json_path}")


def copy_files_from_json(json_path: str, project_root: str, dest_root: str):
    """Copy files to the destination directory according to the file list in json, preserving directory structure."""
    with open(json_path, "r", encoding="utf-8") as f:
        rel_paths: List[str] = json.load(f)

    copied_count = 0
    
    for rel in rel_paths:
        # Check whether it is the unified header special marker
        if rel.startswith("UNIFIED_HEADER:"):
            unified_src = rel.replace("UNIFIED_HEADER:", "")
            # Copy unified header to the include subdir under destination
            unified_dst = os.path.join(dest_root, "include", "unified_definitions.h")
            os.makedirs(os.path.dirname(unified_dst), exist_ok=True)
            shutil.copy2(unified_src, unified_dst)
            print(f"Unified header copied to: include/unified_definitions.h")
            copied_count += 1
        else:
            # Normal files: copy preserving original directory structure
            src = os.path.join(project_root, rel)
            dst = os.path.join(dest_root, rel)
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied_count += 1
    
    print(f"Copied {copied_count} files to {dest_root}")

    # Return the file list, possibly for later batch processing
    return rel_paths


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Config file {config_path} not found; please ensure it exists.")
    
    
    # ---- Modify as needed ----
    DO_COLLECT = True      # generate json
    DO_COPY    = True     # copy files
    
    # Collection strategy selection (choose one)
    COLLECTION_STRATEGY = "full"  # "full", "unified"
    
    
    module_name = "CanMgr"
    project_root = config["ecu_source_dir"]
    entry_file   = config["ecu_entry_file_path"]
    json_path    = f"./output/{module_name}/cTarFiles.json"
    dest_root    = f"./projClosure/{module_name}"

    # ----------------------
    if DO_COLLECT:
        headers = collect_headers(os.path.abspath(entry_file), os.path.abspath(project_root), os.path.abspath(entry_file))
        
        save_headers_to_json(headers, project_root, json_path)

    if DO_COPY:
        _rel_paths = copy_files_from_json(json_path, project_root, dest_root)
