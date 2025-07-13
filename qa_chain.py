from langchain.chains import RetrievalQAWithSourcesChain

def create_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
