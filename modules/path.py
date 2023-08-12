import os

modelfile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/checkpoints/'))
lorafile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/loras/'))
temp_outputs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/'))

os.makedirs(temp_outputs_path, exist_ok=True)
