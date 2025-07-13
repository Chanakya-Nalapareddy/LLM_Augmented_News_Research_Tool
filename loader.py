from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(urls, chunk_size=1000):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    if not data:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=chunk_size,
    )
    docs = text_splitter.split_documents(data)
    return docs
