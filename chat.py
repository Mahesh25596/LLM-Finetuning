import ollama

def chatbot():
    print("Banking Support Bot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = ollama.generate(
            model="bank-support",
            prompt=f"Instruction: Answer this banking customer question\nInput: {user_input}"
        )
        print(f"Bot: {response['response']}")

if __name__ == "__main__":
    chatbot()