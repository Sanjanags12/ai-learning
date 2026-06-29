from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Load existing ChromaDB
print("📂 Loading ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print(f"✅ Loaded {vectorstore._collection.count()} chunks")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_pdf(question):
    # Step 1: Find relevant chunks
    results = vectorstore.similarity_search(question, k=3)

    # Step 2: Build context with sources
    context = ""
    sources = []
    for i, doc in enumerate(results):
        context += f"\n[Source {i+1} - Page {doc.metadata.get('page', '?')}]\n"
        context += doc.page_content + "\n"
        sources.append(f"Page {doc.metadata.get('page', '?')}")

    # Step 3: Ask Groq with context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say 'Not found in document'.

Context:
{context}"""
            },
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    return answer, sources

# Interactive Q&A
print("\n📚 PDF Q&A Bot Ready!")
print("=" * 50)

while True:
    question = input("\n❓ Ask a question (or 'quit'): ")
    if question.lower() == "quit":
        break

    answer, sources = ask_pdf(question)
    print(f"\n🤖 Answer: {answer}")
    print(f"📌 Sources: {', '.join(sources)}")