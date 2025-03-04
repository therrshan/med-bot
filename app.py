from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, clean_chatbot_response, process_user_input
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint

import warnings
from dotenv import load_dotenv
from src.prompt import *
import os
import warnings

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('Pinecone_Key')
HF_API_KEY = os.environ.get('HF_Key')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="med-bot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
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
    app.run(host="0.0.0.0", port= 8080, debug= True)