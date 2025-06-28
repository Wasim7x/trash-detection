import os
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parent.parent)

def __init__(self):
    pass
    
class CommonFun:
    @staticmethod
    def create_dir(path):
       """Create a directory if it does not exist."""
       if not os.path.exists(path):
           os.makedirs(path)

    @staticmethod
    def get_absolute_path(relative_path: str) -> str: 
        """
        Get the absolute path of a given relative path
        """      
        return str(Path(__file__).resolve().parent.parent / relative_path)  
    
    @staticmethod
    def write_to_yaml(file_path: str, data: dict):
        """
        Write a dictionary to a YAML file.
        """
        import yaml
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

           