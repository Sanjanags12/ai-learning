import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB (saves locally — no API key!)
client = chromadb.Client()
collection = client.create_collection("fitness_knowledge")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Our fitness knowledge base
documents = [
    "Squats build strong legs and glutes",
    "Bench press develops chest and triceps strength",
    "Deadlifts work the entire posterior chain",
    "Pull ups are best for back and biceps",
    "Plank strengthens core and improves posture",
    "Eating 1g protein per pound of bodyweight builds muscle",
    "Caloric deficit of 500 calories loses 0.5kg per week",
    "HIIT cardio burns more fat than steady state cardio",
    "Sleep 7-9 hours for optimal muscle recovery",
    "Drink 3-4 litres of water daily for best performance",
]

# Add to ChromaDB
embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("✅ Database created with", collection.count(), "documents")
print()

# Query the database
queries = [
    "How do I get bigger arms?",
    "What should I eat to lose weight?",
    "How to recover faster after gym?",
]

for query in queries:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    print(f"🔍 Query: {query}")
    for doc in results['documents'][0]:
        print(f"   → {doc}")
    print()