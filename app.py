import os
import time
import streamlit as st

from config import FAISS_STORE_PATH, NUM_URLS
from loader import load_and_split_documents
from model import load_llm, get_embeddings
from vectorstore import create_vectorstore, save_vectorstore, load_vectorstore
from qa_chain import create_qa_chain

# Initialize LLM once at startup
llm = load_llm()
embeddings = get_embeddings()

st.title("📰 News Research Tool")
st.sidebar.title("Enter News URLs")

urls = []
for i in range(NUM_URLS):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("🔄 Loading and extracting data from URLs...")
    try:
        docs = load_and_split_documents(urls)
        if not docs:
            st.error("⚠️ No data could be loaded from the URLs. Please check the links.")
        else:
            main_placeholder.text("🧠 Creating vector store...")
            vectorstore = create_vectorstore(docs, embeddings)
            time.sleep(2)
            save_vectorstore(vectorstore, FAISS_STORE_PATH)
            st.success("✅ Vector store created and saved successfully.")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

query = st.text_input("Ask a question about the articles:")

if query:
    if os.path.exists(FAISS_STORE_PATH):
        vectorstore = load_vectorstore(FAISS_STORE_PATH)
        chain = create_qa_chain(llm, vectorstore)
        with st.spinner("🔍 Generating answer..."):
            result = chain({"question": query}, return_only_outputs=True)
        st.header("🧾 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("📚 Sources")
            st.write(sources)
    else:
        st.error("⚠️ Please process URLs first to create the vector store.")
