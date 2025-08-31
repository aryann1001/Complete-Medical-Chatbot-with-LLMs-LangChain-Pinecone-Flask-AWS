# src/helper.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# 📌 1. Document Loading (for local ingestion)
# ============================================================
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# ============================================================
# 📌 2. HuggingFace Embeddings
# ============================================================
def get_embeddings():
    """Shared embedding function for both ingestion & retrieval"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


# ============================================================
# 📌 3. Pinecone Retrieval
# ============================================================
def get_pinecone_vectorstore(index_name: str):
    embeddings = get_embeddings()   # ✅ now always provided
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
