LLM Fine-Tuning Project for Banking Support Chatbot
==================================================

Project Description:
This project fine-tunes a Mistral language model using Ollama to create a banking customer support chatbot. The model is trained on banking FAQ data to answer common customer questions.

Prerequisites:
1. Python 3.8+
2. Ollama installed and running
3. Required Python packages: ollama, pandas, scikit-learn

Setup:
1. Clone this repository
2. Create virtual environment:
   python -m venv llm-tuning
3. Activate environment:
   - Windows: llm-tuning\Scripts\activate
   - Mac/Linux: source llm-tuning/bin/activate
4. Install requirements:
   pip install ollama pandas scikit-learn

Project Structure:
- prepare_data.py: Prepares and formats training data
- fine_tune.py: Creates and fine-tunes the model
- evaluate.py: Evaluates model performance
- chat.py: Interactive chatbot interface
- train.jsonl: Training data
- val.jsonl: Validation data

Data Sources:
Banking FAQ dataset from:
https://www.kaggle.com/datasets/manojajj/banking-chatbot?resource=download 

Usage:
1. Prepare data:
   python prepare_data.py
2. Fine-tune model:
   python fine_tune.py
3. Evaluate model:
   python evaluate.py
4. Chat with model:
   python chat.py

Troubleshooting:
- If you get encoding errors, try converting CSV files to UTF-8
- Ensure Ollama is running (ollama serve)
- For model creation issues, verify base model is pulled:
  ollama pull mistral

Notes:
- The current implementation uses parameter-efficient fine-tuning
- Full fine-tuning requires more resources
- Model performance depends on quality and quantity of training data

License:
MIT License - Free for academic and commercial use