# src/helper.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_vectorstore(index_name: str):
    """
    Connect to an existing Pinecone index and return a VectorStore.
    NOTE: Embeddings are not created here (only used at ingestion).
    """
    # Use a "dummy" embedding function — not used for runtime similarity search
    # because Pinecone already stores embeddings. We just need retrieval.
    from langchain_core.embeddings import Embeddings

    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            raise NotImplementedError("Not used in runtime retrieval")

        def embed_query(self, text):
            raise NotImplementedError("Not used in runtime retrieval")

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=DummyEmbeddings()
    )
