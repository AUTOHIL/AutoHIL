import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Output file path
module_name = "<your_module_name>" 
OUTPUT_FILE = f"./output/{module_name}/summary/function_info_map.json"

# Input configuration method: choose one of the following:
# Option 1: auto search mode - recursively find all function_info_map.json files
SEARCH_DIRECTORY = f"./output/{module_name}/summary"  # search directory
# Option 2: explicit file list - if the list is non-empty, use it; otherwise, use directory search
INPUT_FILES = [
    f"./output/{module_name}/summary/function_info_map1.json",
    f"./output/{module_name}/summary/function_info_map2.json"
]


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file
    
    Args:
        file_path: JSON file path
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return {}


def is_function_info_complete(func_info: Dict[str, Any]) -> bool:
    """
    Check whether function info is complete (file, lineNumber, lineNumberEnd are all non-empty)
    
    Args:
        func_info: function info dict
        
    Returns:
        True if info is complete, False otherwise
    """
    file_path = func_info.get('file', '')
    line_number = func_info.get('lineNumber', 0)
    line_number_end = func_info.get('lineNumberEnd', 0)
    
    # Check file is a non-empty string, and lineNumber/lineNumberEnd are positive
    return (
        isinstance(file_path, str) and file_path.strip() != '' and
        isinstance(line_number, (int, float)) and line_number > 0 and
        isinstance(line_number_end, (int, float)) and line_number_end > 0
    )


def choose_better_function_info(existing_info: Dict[str, Any], new_info: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """
    Choose the better function info
    
    Args:
        existing_info: existing function info
        new_info: new function info
        
    Returns:
        (chosen function info, reason)
    """
    existing_complete = is_function_info_complete(existing_info)
    new_complete = is_function_info_complete(new_info)
    
    if existing_complete and not new_complete:
        return existing_info, "Keep existing (more complete info)"
    elif not existing_complete and new_complete:
        return new_info, "Use new version (more complete info)"
    elif existing_complete and new_complete:
        # If both are complete, prefer .c files
        existing_file = existing_info.get('file', '').lower()
        new_file = new_info.get('file', '').lower()
        
        existing_is_c = existing_file.endswith('.c')
        new_is_c = new_file.endswith('.c')
        
        if existing_is_c and not new_is_c:
            return existing_info, "Keep existing (.c file preferred)"
        elif not existing_is_c and new_is_c:
            return new_info, "Use new version (.c file preferred)"
        elif existing_is_c and new_is_c:
            return new_info, "Use new version (both are .c files, overwrite)"
        else:
            return existing_info, "Keep existing (neither is a .c file; keep the first)"
    else:
        return existing_info, "Keep existing (both incomplete; keep the first)"


def merge_function_maps() -> None:
    """
    Merge multiple function_info_map files
    """
    # Get the input file list
    if INPUT_FILES:  # if the list is non-empty
        input_files = INPUT_FILES
        print(f"Using explicit file list mode")
    else:
        input_files = find_function_map_files(SEARCH_DIRECTORY)
        print(f"Using directory search mode: {SEARCH_DIRECTORY}")
    
    if not input_files:
        print("No function_info_map.json files found")
        return
    
    merged_data = {}
    conflict_count = 0
    total_functions = 0
    replacement_count = 0
    
    for i, file_path in enumerate(input_files, 1):
        if not os.path.exists(file_path):
            print(f"File does not exist, skipping")
            continue
            
        data = load_json_file(file_path)
        if not data:
            continue
            
        file_functions = 0
        file_conflicts = 0
        file_replacements = 0
        
        for func_name, func_info in data.items():
            file_functions += 1
            
            if func_name in merged_data:
                file_conflicts += 1
                conflict_count += 1
                
                # Use intelligent selection strategy
                chosen_info, reason = choose_better_function_info(merged_data[func_name], func_info)
                
                if chosen_info is func_info:  # if the new function info was chosen
                    merged_data[func_name] = func_info
                    file_replacements += 1
                    replacement_count += 1
                    print(f"  Function conflict: {func_name} → {reason}")
                else:
                    print(f"  Function conflict: {func_name} → {reason}")
            else:
                merged_data[func_name] = func_info
    
        total_functions += file_functions
    
    if not merged_data:
        print("No function data was successfully read")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Write the merged output file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("Merge completed!")
        print("=" * 60)
        print(f"Statistics:")
        print(f"   Total functions processed: {total_functions}")
        print(f"   Functions after merge: {len(merged_data)}")
        print(f"   Conflicts found: {conflict_count}")
        print(f"   Intelligent replacements: {replacement_count}")
        print(f"   Conflict resolution strategy: intelligent selection (prefer complete info)")
        print(f"   Output file: {OUTPUT_FILE}")
        print(f"   File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"Failed to write output file {OUTPUT_FILE}: {e}")


def find_function_map_files(directory: str) -> List[str]:
    """
    Find all function_info_map.json files under the specified directory
    
    Args:
        directory: search directory
        
    Returns:
        List of found files
    """
    files = []
    search_path = Path(directory)
    
    if not search_path.exists():
        print(f"Search directory does not exist: {directory}")
        return files
    
    for file_path in search_path.rglob('function_info_map[0-9].json'):
        files.append(str(file_path))
    
    return files


def main():
    print("=" * 60)
    print("Function Info Map Merge Tool - Full Version")
    print("=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Conflict resolution strategy: intelligent selection (prefer complete info)")
    
    if INPUT_FILES:  # if the list is non-empty
        print("Input mode: explicit file list")
        print("File list:")
        for f in INPUT_FILES:
            print(f" {f}")
    else:
        print(f"Input mode: directory search ({SEARCH_DIRECTORY})")
    
    print("-" * 60)
    
    merge_function_maps()


if __name__ == '__main__':
    main()
