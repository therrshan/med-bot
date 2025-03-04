# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01 - Install the requirements
```bash
pip install -r requirements.txt
```


### STEP 02 -Create a `.env` file in the root directory and add your Pinecone & Hugging Face credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HUGGINGFACEHUB_API_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
