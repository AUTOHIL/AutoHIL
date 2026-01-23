import pandas as pd
import os
import sys
from typing import List, Dict, Any, Optional


def save_results_to_excel(data_list, filename):
    if not data_list:
        return
    try:
        df = pd.DataFrame(data_list)
        df.to_excel(filename, index=False, engine="openpyxl")
    except Exception as e:
        print(f"Fail to save file {filename}: {str(e)}")


class ExcelParser:
    """Excel file parser for extracting specific columns' data"""
    
    def __init__(self, file_path: str, sheet_name="Sheet1"):
        """
        Initialize parser

        Args:
            file_path: path to the Excel file
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.data = None
        
    def load_excel(self) -> bool:
        """
        Load Excel file

        Returns:
            bool: whether loading succeeded
        """
        try:
            if not os.path.exists(self.file_path):
                return False
                
            # Try to read the specified Excel sheet
            self.data = pd.read_excel(self.file_path, self.sheet_name)
            
            return True
            
        except Exception as e:
            return False
    
    def extract_target_data(self, target_columns) -> List[Dict[str, Any]]:
        """
        Extract all data from the target columns

        Args:
            sheet_name: specified sheet name; if None, auto-detect

        Returns:
            List[Dict]: list of dicts containing row indices and data
        """
        if self.data is None:
            return []
            
        # Extract target columns' data
        target_data = []
        pri_col = target_columns[0]  # use req_id as baseline to determine empty rows
        for index, row in self.data.iterrows():
            req_id = row[pri_col]

            if pd.notna(req_id) and str(req_id).strip():  # filter out NaN and empty strings
                row_data = {}
                for col in target_columns:
                    value = row[col]
                    row_data[col] = str(value).strip() if pd.notna(value) else None

                target_data.append({
                    'row_index': index + 2,  # Excel row number (starts from row 2 because row 1 is header)
                    'row_data': row_data
                })

        return target_data
    
    def save_extracted_data(self, data: List[Dict[str, Any]], target_columns: List[str], output_file: str = None) -> bool:
        """
        Save the extracted data to a file

        Args:
            data: extracted data
            output_file: output file path; if None, generated automatically

        Returns:
            bool: whether saving succeeded
        """
        if not data:
            return False
            
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_file = f"Requirement_augmentation_data.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"'Requirement augmentation' data extracted from file {self.file_path}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, item in enumerate(data, 1):
                    f.write(f"{i}. Row number: {item['row_index']}\n")
                    for col_name in target_columns:
                        value = item['row_data'].get(col_name, 'N/A')
                        f.write(f"  {col_name}: {value}\n")
                    f.write("\n")
            
            print(f"Data has been saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save data: {str(e)}")
            return False
