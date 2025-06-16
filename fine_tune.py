import ollama
import json
import time
from pathlib import Path
import subprocess
import sys

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def create_model(model_name, modelfile_content):
    """Helper function to create a model with error handling"""
    # Save Modelfile
    modelfile_path = f"Modelfile_{model_name}"
    Path(modelfile_path).write_text(modelfile_content, encoding='utf-8')
    
    print(f"\nCreating model: {model_name}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if result.returncode != 0:
        print(f"Error creating {model_name}:")
        print(result.stderr)
        return False
    
    print(f"Successfully created {model_name}")
    print(result.stdout)
    return True

# Load training data
train_data = load_data('train.jsonl')

# 1. Create basic model - SIMPLIFIED CORRECT SYNTAX
base_model = "mistral"
custom_model = "bank-support"

# CORRECTED Modelfile syntax
basic_modelfile = f"""FROM {base_model}
SYSTEM You are a helpful banking customer support assistant.
"""
# Removed PARAMETER lines as they were causing syntax errors

if not create_model(custom_model, basic_modelfile):
    sys.exit(1)

# 2. Test the model
print("\nTesting model...")
test_questions = [
    "How do I reset my online banking password?",
    "What's your customer service phone number?",
    "How do I open a new account?"
]

for question in test_questions:
    print(f"\nQ: {question}")
    try:
        response = ollama.generate(model=custom_model, prompt=question)
        print("Response:", response['response'])
    except Exception as e:
        print(f"Error: {str(e)}")

print("\nProcess completed!")