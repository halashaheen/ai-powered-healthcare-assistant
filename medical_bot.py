from groq import Groq

client = Groq(api_key="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")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a medical education assistant helping students learn. Always remind users you're for educational purposes only."},
            {"role": "user", "content": user_input}
        ]
    )
    
    print(f"Bot: {response.choices[0].message.content}\n")