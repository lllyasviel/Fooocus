import json

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage: load_prompts("jsonAF/prompts.json")