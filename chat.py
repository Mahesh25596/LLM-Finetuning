import ollama
import sys

def check_model_available(model_name):
    """Check if the specified model is available"""
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        return any(model_name in name for name in available_models)
    except:
        return False

def chatbot():
    print("Banking Support Chatbot (Fine-Tuned)")
    print("Type 'quit' to exit\n")
    
    # Determine which model to use
    if check_model_available('bank-support-finetuned'):
        model_name = 'bank-support-finetuned'
        print("Using: Fine-tuned banking model")
    elif check_model_available('bank-support-enhanced'):
        model_name = 'bank-support-enhanced'
        print("Using: Enhanced context banking model")
    else:
        model_name = 'mistral'
        print("Using: Base Mistral model (no fine-tuning detected)")
    
    print("-" * 40)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("Customer: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for using our banking support!")
                break
                
            if not user_input:
                continue
            
            # Generate response
            response = ollama.generate(
                model=model_name,
                prompt=user_input,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent responses
                    'top_p': 0.9,
                    'num_predict': 500
                }
            )
            
            bot_response = response['response'].strip()
            print(f"Support: {bot_response}")
            
            # Store conversation
            conversation_history.append(f"Customer: {user_input}")
            conversation_history.append(f"Support: {bot_response}")
            
        except KeyboardInterrupt:
            print("\n\nSession ended by user.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    chatbot()
