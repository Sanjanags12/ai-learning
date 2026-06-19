from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = "Write a creative tagline for a fitness app"

print("=" * 50)
print("TEMPERATURE EXPERIMENT")
print("=" * 50)

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=temp
    )
    print(f"\n🌡 Temperature {temp}:")
    print(response.choices[0].message.content)
    print("-" * 40)