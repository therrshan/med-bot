import warnings
import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, clean_chatbot_response, process_user_input
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaLLM
from src.prompt import *

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HF_API_KEY = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
if not HF_API_KEY:
    raise ValueError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN not found in environment variables")

# Set Pinecone key
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# DON'T SET HF TOKEN YET - wait until after embeddings load
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Load embeddings (helper.py will handle token removal/restoration)
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="med-bot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OllamaLLM(
    model="llama3",
    temperature=0.7,
)

prompt_format = get_prompt_format()
question_answer_chain = create_stuff_documents_chain(llm, prompt_format)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = process_user_input(msg)
    print(input)
    response = rag_chain.invoke({"input": input})
    print("Response : ", clean_chatbot_response(response))
    return str(clean_chatbot_response(response))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)