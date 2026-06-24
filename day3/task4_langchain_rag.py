from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
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

# Step 5: RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("✅ RAG Pipeline Ready!\n")
print("=" * 50)

questions = [
    "What is the main topic of this document?",
    "What problem does this paper solve?",
    "What are the key findings?",
]

for q in questions:
    print(f"\n❓ {q}")
    result = qa_chain.invoke({"query": q})
    print(f"🤖 {result['result']}")
    print(f"📌 Sources: {len(result['source_documents'])} chunks used")
    print("-" * 50)