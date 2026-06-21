import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("fitness_kb")

# Knowledge base
documents = [
    "For fat loss: eat in 500 calorie deficit, do cardio 3-4x per week",
    "For muscle gain: eat in 300 calorie surplus, lift heavy 4-5x per week",
    "Best pre-workout meal: banana + peanut butter 45 mins before",
    "Best post-workout meal: protein shake + rice within 30 mins",
    "Squats, deadlifts and bench press are the 3 best compound exercises",
    "Sleep 8 hours minimum for muscle growth and fat loss",
    "Drink water before meals to reduce hunger and aid fat loss",
    "Progressive overload: increase weight each week to keep growing",
]

# Store in ChromaDB
embeddings = embed_model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("🏋️ AI Fitness Trainer Ready!")
print("=" * 50)

while True:
    user_question = input("\nAsk your trainer (or 'quit'): ")
    if user_question.lower() == "quit":
        break

    # Step 1: Find relevant knowledge
    query_embedding = embed_model.encode([user_question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    context = "\n".join(results['documents'][0])

    # Step 2: Send to Groq with context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert fitness trainer.
Use ONLY this knowledge to answer:

{context}

Be specific, practical, and encouraging."""
            },
            {"role": "user", "content": user_question}
        ]
    )

    print(f"\n🤖 Trainer: {response.choices[0].message.content}")