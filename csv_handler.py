# csv_handler.py
import pandas as pd
from typing import Tuple, Optional, Dict, Any

class CSVHandler:
    def __init__(self):
        self.df = None
        self.file_info = {}
        
    def load_csv(self, file_obj) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Load and validate a CSV file
        
        Args:
            file_obj: File object from Gradio upload
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Error message if any (str or None)
            - File info dictionary (dict)
        """
        try:
            # Read the CSV into a pandas DataFrame
            self.df = pd.read_csv(file_obj)
            
            # Generate file info
            self.file_info = {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "column_names": list(self.df.columns),
                "data_types": {col: str(self.df[col].dtype) for col in self.df.columns},
                "sample_data": self.df.head(5).to_dict()
            }
            
            return True, None, self.file_info
            
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}", {}
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the loaded DataFrame"""
        return self.df
    
    def get_column_stats(self, column_name: str) -> Dict[str, Any]:
        """Get statistics for a specific column"""
        if self.df is None or column_name not in self.df.columns:
            return {"error": "Invalid column or no data loaded"}
        
        col = self.df[column_name]
        stats = {}
        
        if pd.api.types.is_numeric_dtype(col):
            stats = {
                "min": col.min(),
                "max": col.max(),
                "mean": col.mean(),
                "median": col.median(),
                "std": col.std()
            }
        else:
            # For non-numeric columns
            stats = {
                "unique_values": col.nunique(),
                "most_common": col.value_counts().head(5).to_dict()
            }
            
        return stats