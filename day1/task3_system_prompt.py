from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = "Should I eat before my workout?"

# WITHOUT system prompt
print("=" * 50)
print("WITHOUT System Prompt:")
print("=" * 50)

response1 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": question}
    ]
)
print(response1.choices[0].message.content)

# WITH system prompt
print("\n" + "=" * 50)
print("WITH System Prompt (Fitness Coach):")
print("=" * 50)

response2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are an expert fitness coach with 10 years experience. Give practical, science-based advice. Always mention timing, food types, and portion size."
        },
        {"role": "user", "content": question}
    ]
)
print(response2.choices[0].message.content)