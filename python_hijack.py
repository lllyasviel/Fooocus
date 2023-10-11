import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)
backend_path = os.path.join(root, 'backend', 'headless')
if backend_path not in sys.path:
    sys.path.append(backend_path)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
