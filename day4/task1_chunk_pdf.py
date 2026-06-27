from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("document.pdf")
pages = loader.load()

print(f" Loaded {len(pages)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 500 characters per chunk
    chunk_overlap=50,     # 50 chars overlap between chunks
)

chunks = splitter.split_documents(pages)

print(f" Split into {len(chunks)} chunks")
print(f"\n First chunk:")
print(chunks[0].page_content)
print(f"\n Source: Page {chunks[0].metadata['page']}")