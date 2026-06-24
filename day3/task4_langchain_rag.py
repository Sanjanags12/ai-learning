from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

print("🔧 Building RAG Pipeline...")

# Step 1: Load PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages")

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks created")

# Step 3: Embed + Store
print("🔢 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Stored in ChromaDB")

# Step 4: LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

print("✅ RAG Pipeline Ready!\n")
print("=" * 50)

def ask(question):
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Ask LLM
    messages = [
        {"role": "system", "content": f"""Answer using ONLY this context:
{context}

If answer not in context say 'Not found in document'."""},
        {"role": "user", "content": question}
    ]
    
    response = llm.invoke(messages)
    sources = [f"Page {doc.metadata.get('page', '?')}" for doc in docs]
    return response.content, sources

# Ask questions
questions = [
    "What is the main topic of this document?",
    "What problem does this paper solve?",
    "What are the key findings?",
]

for q in questions:
    print(f"\n❓ {q}")
    answer, sources = ask(q)
    print(f"🤖 {answer}")
    print(f"📌 Sources: {', '.join(sources)}")
    print("-" * 50)