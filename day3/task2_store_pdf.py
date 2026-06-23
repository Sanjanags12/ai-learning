from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("📄 Loading PDF...")
loader = PyPDFLoader("document.pdf")
pages = loader.load()

print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks created")

print("🔢 Creating embeddings + storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # free, no API key!
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # saves locally
)

print(f"✅ Stored {vectorstore._collection.count()} chunks in ChromaDB!")
print("📁 Database saved to ./chroma_db folder")

# Test a quick search
query = "What is retrieval augmented generation?"
results = vectorstore.similarity_search(query, k=3)

print(f"\n🔍 Test Query: {query}")
print("\nTop 3 Results:")
for i, doc in enumerate(results):
    print(f"\n{i+1}. Page {doc.metadata.get('page', '?')}:")
    print(f"   {doc.page_content[:150]}...")