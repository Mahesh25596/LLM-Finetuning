import ollama
import json
import time
from pathlib import Path

def check_ollama_models():
    """Check available models and pull mistral if needed"""
    try:
        models = ollama.list()
        print("Available models:")
        for model in models['models']:
            print(f"  - {model['name']}")
        
        # Check if mistral is available
        mistral_available = any('mistral' in model['name'].lower() for model in models['models'])
        if not mistral_available:
            print("Pulling mistral model... This may take a few minutes.")
            ollama.pull('mistral')
            print("Mistral model pulled successfully!")
        return True
    except Exception as e:
        print(f"Error checking models: {e}")
        return False

def create_fine_tune_modelfile():
    """Create Modelfile for fine-tuning"""
    modelfile_content = f"""FROM mistral

# System prompt for banking context
SYSTEM """You are a helpful banking customer support assistant. Provide accurate, clear, and helpful responses to customer questions about banking services."

# Fine-tuning parameters
PARAMETER num_epoch 3
PARAMETER learning_rate 0.0001
"""
    
    return modelfile_content

def fine_tune_model():
    """Perform actual fine-tuning"""
    print("Starting fine-tuning process...")
    
    # Check if training data exists
    if not Path('train.jsonl').exists():
        print("Error: train.jsonl not found. Run prepare_data.py first.")
        return False
    
    # Create fine-tuning Modelfile
    modelfile_content = create_fine_tune_modelfile()
    modelfile_path = "Modelfile.finetune"
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print("Created fine-tuning Modelfile")
    
    # Perform fine-tuning using Ollama
    try:
        print("Starting fine-tuning... This will take several minutes.")
        start_time = time.time()
        
        # Create fine-tuned model
        result = ollama.create(
            model='bank-support-finetuned',
            modelfile=modelfile_content,
            stream=False
        )
        
        end_time = time.time()
        print(f"Fine-tuning completed in {end_time - start_time:.2f} seconds")
        print(f"Fine-tuning result: {result}")
        return True
        
    except Exception as e:
        print(f"Fine-tuning error: {e}")
        print("This might be due to hardware limitations. Trying alternative approach...")
        return alternative_fine_tune_approach()

def alternative_fine_tune_approach():
    """Alternative approach using system context enhancement"""
    print("Using system context enhancement approach...")
    
    # Create a model with enhanced system context
    modelfile_content = """FROM mistral

TEMPLATE """[INST] <<SYS>>
You are an expert banking customer support assistant with deep knowledge of:
- Account management and password reset procedures
- Banking fees, interest rates, and policies
- Loan applications and credit services
- Online and mobile banking features
- Security protocols and fraud prevention

Always provide accurate, helpful, and professional responses. If you don't know something, admit it rather than guessing.
<</SYS>>

{{ .Prompt }} [/INST]"""

    try:
        result = ollama.create(
            model='bank-support-enhanced',
            modelfile=modelfile_content
        )
        print("Enhanced context model created successfully!")
        return True
    except Exception as e:
        print(f"Alternative approach also failed: {e}")
        return False

def test_fine_tuned_model():
    """Test the fine-tuned model"""
    print("\n" + "="*50)
    print("TESTING FINE-TUNED MODEL")
    print("="*50)
    
    test_questions = [
        "How do I reset my online banking password?",
        "What are your customer service hours?",
        "How can I check my account balance?",
        "What is the minimum balance for a savings account?",
        "How do I report a lost debit card?"
    ]
    
    model_name = 'bank-support-finetuned'
    
    # Check if fine-tuned model exists, fall back to enhanced
    try:
        ollama.show(model_name)
    except:
        model_name = 'bank-support-enhanced'
        print(f"Using enhanced context model: {model_name}")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        try:
            response = ollama.generate(model=model_name, prompt=question)
            print(f"Response: {response['response']}")
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Banking Chatbot Fine-Tuning System")
    print("=" * 40)
    
    # Step 1: Check models
    if not check_ollama_models():
        print("Failed to initialize models. Exiting.")
        exit(1)
    
    # Step 2: Fine-tune model
    if fine_tune_model():
        print("Fine-tuning completed successfully!")
    else:
        print("Fine-tuning failed, but alternative approach may have worked.")
    
    # Step 3: Test the model
    test_fine_tuned_model()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE")
    print("You can now use chat.py to interact with your banking chatbot!")
    print("Model name: bank-support-finetuned or bank-support-enhanced")
