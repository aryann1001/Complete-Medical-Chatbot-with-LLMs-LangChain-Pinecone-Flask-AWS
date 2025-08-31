import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_vectorstore(index_name: str):
    """
    Connect to Pinecone index and return a VectorStore.
    Uses HuggingFace embeddings for query similarity.
    """
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed_model
    )
