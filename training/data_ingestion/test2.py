import os
import sys
from pathlib import Path
sys.path[0]=str(Path(__file__).resolve().parent.parent)
os.makedirs("training\\artifacts\\temp",exist_ok=False)