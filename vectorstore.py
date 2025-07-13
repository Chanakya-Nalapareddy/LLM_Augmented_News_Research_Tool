import pickle
from langchain_community.vectorstores import FAISS

def create_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

def save_vectorstore(vectorstore, path):
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(path):
    with open(path, "rb") as f:
        return pickle.load(f)
