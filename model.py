import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

def load_llm(model_id="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return HuggingFacePipeline(pipeline=pipe)

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
