from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This simulates a real conversation with memory
messages = [
    {
        "role": "system",
        "content": "You are a personal AI fitness trainer. Remember what the user tells you and give personalized advice."
    }
]

print("🏋️ AI Fitness Trainer — Type 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Goodbye! Stay consistent! 💪")
        break

    # Add user message to history
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Get response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    reply = response.choices[0].message.content

    # Add AI reply to history (so it remembers)
    messages.append({
        "role": "assistant",
        "content": reply
    })

    print(f"\n🏋️ AI Trainer: {reply}")