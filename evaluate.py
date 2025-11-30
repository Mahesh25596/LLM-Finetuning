import ollama
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

def load_test_data():
    """Load validation data for evaluation"""
    with open('val.jsonl', 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def evaluate_model(model_name, test_data, sample_size=10):
    """Evaluate model performance"""
    print(f"Evaluating model: {model_name}")
    print(f"Test samples: {min(sample_size, len(test_data))}")
    
    scores = []
    predictions = []
    
    for i, item in enumerate(test_data[:sample_size]):
        # Extract the user question from the formatted text
        text = item['text']
        user_part = text.split('<|im_start|>assistant')[0].replace('<|im_start|>user\n', '').replace('<|im_end|>', '')
        
        # Get model prediction
        try:
            response = ollama.generate(model=model_name, prompt=user_part.strip())
            prediction = response['response'].strip()
            predictions.append(prediction)
            
            # Simple scoring based on response quality
            score = min(len(prediction) / 100, 1.0)  # Basic heuristic
            scores.append(score)
            
            print(f"\nSample {i+1}:")
            print(f"Question: {user_part[:100]}...")
            print(f"Response: {prediction[:100]}...")
            print(f"Score: {score:.2f}")
            
        except Exception as e:
            print(f"Error on sample {i+1}: {e}")
            predictions.append("")
            scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS for {model_name}")
    print(f"{'='*50}")
    print(f"Average Score: {avg_score:.2f}/1.0")
    print(f"Response Rate: {len([s for s in scores if s > 0])}/{len(scores)}")
    
    return avg_score

if __name__ == "__main__":
    # Test different model versions
    models_to_test = ['bank-support-finetuned', 'bank-support-enhanced', 'mistral']
    
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test samples")
    
    results = {}
    for model_name in models_to_test:
        try:
            ollama.show(model_name)
            score = evaluate_model(model_name, test_data)
            results[model_name] = score
        except:
            print(f"Model {model_name} not available, skipping...")
    
    print("\nFINAL COMPARISON:")
    for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {score:.2f}")
