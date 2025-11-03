from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    # Save and remove any existing tokens
    saved_tokens = {}
    token_keys = ['HUGGINGFACEHUB_API_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']
    
    for key in token_keys:
        if key in os.environ:
            saved_tokens[key] = os.environ.pop(key)
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
    finally:
        # Restore tokens
        for key, value in saved_tokens.items():
            os.environ[key] = value
    
    return embeddings

def process_user_input(user_input):
    if not user_input.endswith('?'):
        user_input += '?'
    return user_input

def clean_chatbot_response(response):
    cleaned_response = response["answer"].replace("Assistant:", "").strip()
    return cleaned_response