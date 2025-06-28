import os
from pathlib import Path
import sys
import yaml
sys.path[0] = str(Path(__file__).resolve().parent.parent)
from utils.file_counter import FileCounter
destination_path = Path(Path(__file__).resolve().parent.parent.parent,"garbage-dataset")
print(f"Destination path: {destination_path}")

no_files = FileCounter.count_files(str(destination_path))
print(f"Number of files : {no_files}") 


config_file = r"training\\configs\\config.yaml"
with open(config_file, 'w+') as f:
    yaml.safe_dump({"output_path": str(destination_path)}, f, default_flow_style=False)
print("successfully updated config ")    
# sys.path[0] = str(Path(__file__).resolve().parent.parent)

# print(str(Path(__file__).resolve().parent.parent.parent))