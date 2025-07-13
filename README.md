# 🧠 LLM-Augmented News Research Tool

**LLM-Augmented News Research Tool** is an intelligent system designed to analyze online news articles using a large language model (LLM). The tool extracts and processes content from URLs, generates semantic vector embeddings, and enables context-based question answering with relevant sources using Retrieval-Augmented Generation (RAG).

---

## 📁 Project Structure

```
LLM_Augmented_News_Research_Tool/
│
├── app.py
├── loader.py         # Extracts and loads content from URLs
├── config.py         # Environment variables and model setup
├── model.py          # Loads HuggingFace LLM pipeline
├── vectorstore.py    # Text splitting and FAISS vector storage
├── qa_chain.py       # LLM-based RetrievalQA pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Chanakya-Nalapareddy/LLM_Augmented_News_Research_Tool.git
cd LLM_Augmented_News_Research_Tool
```

### 2. Create a New Conda Environment

```bash
conda create -n news_rag python=3.10 -y
conda activate news_rag
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file and include API keys or config variables as needed.

---

## 🚀 Running the App

```bash
python main.py
```

- Enter up to 3 news URLs.
- The tool will extract, chunk, and embed the content.
- Ask questions about the news content and get accurate answers with sources.

---

## 🧩 Dependencies

- `transformers`
- `torch`
- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `sentence-transformers`
- `faiss-cpu`
- `unstructured`
- `python-dotenv`

---

## 📝 License

MIT License. See `LICENSE` file for details.
