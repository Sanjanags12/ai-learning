from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base — fitness facts
knowledge_base = [
    "Squats are great for building leg muscles",
    "You need protein to build muscle after workout",
    "Cardio helps burn fat and improve heart health",
    "Sleep is important for muscle recovery",
    "Drinking water helps with weight loss",
    "Push ups work chest triceps and shoulders",
    "Running burns more calories than walking",
    "Eating less calories than you burn causes weight loss",
]

# User query
query = "How do I lose fat?"

# Embed everything
kb_embeddings = model.encode(knowledge_base)
query_embedding = model.encode([query])

# Find similarity scores
scores = cosine_similarity(query_embedding, kb_embeddings)[0]

# Get top 3 results
top_indices = np.argsort(scores)[::-1][:3]

print(f"🔍 Query: {query}")
print(f"\n🎯 Top 3 Most Similar Results:")
print("-" * 50)
for i, idx in enumerate(top_indices):
    print(f"{i+1}. Score: {scores[idx]:.3f}")
    print(f"   {knowledge_base[idx]}")
    print()
    