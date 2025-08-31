# ingest.py
import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, filter_to_minimal_docs, get_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Make sure index exists
index_name = "medical-chatbot"
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,     # Must match embedding model dimension
        metric="cosine"
    )

# 1. Load PDFs
docs = load_pdf_file("data/")   # assumes PDFs in ./data
print(f"Loaded {len(docs)} documents")

# 2. Split into chunks
chunks = text_split(docs)
print(f"Split into {len(chunks)} chunks")

# 3. Minimal docs (optional, keeps metadata light)
chunks = filter_to_minimal_docs(chunks)

# 4. Embeddings
embeddings = get_embeddings()

# 5. Push to Pinecone
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
vectorstore.add_documents(chunks)

print("✅ Ingestion complete! Data uploaded to Pinecone.")
