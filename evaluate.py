import ollama
import random
import json

# Load some validation examples
with open('val.jsonl', 'r') as f:
    val_examples = [json.loads(line) for line in f]

# Test with some sample questions
def test_model(question, model="bank-support"):
    response = ollama.generate(
        model=model,
        prompt=f"Instruction: Answer this banking customer question\nInput: {question}"
    )
    return response['response']

# Automated evaluation
correct = 0
for example in random.sample(val_examples, 10):  # Test on 10 random examples
    answer = test_model(example['input'])
    print(f"Q: {example['input']}")
    print(f"Expected: {example['output']}")
    print(f"Actual: {answer}\n")
    
    # Simple evaluation - check if key terms are present
    key_terms = set(example['output'].lower().split()[:5])  # first few words
    if any(term in answer.lower() for term in key_terms):
        correct += 1

print(f"Accuracy: {correct/10:.1%}")