from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "I want to lose weight fast",
    "Help me burn fat quickly",
    "I need to build muscle",
    "What should I eat for dinner",
    "Give me a chest workout",
    "How to get six pack abs",
]

# Convert text → numbers (embeddings)
embeddings = model.encode(sentences)

print(" Embeddings created!")
print(f"Shape: {embeddings.shape}")
print(f"Each sentence = {embeddings.shape[1]} numbers")
print(f"\nFirst embedding (first 5 numbers):")
print(embeddings[0][:5])