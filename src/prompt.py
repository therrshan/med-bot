from langchain.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Greet the user if the user greets you properly and respectfully without considering the context"
    "When a user asks any question, use the provided retrieved context to answer the question. "
    "If you don't know, say so. Keep responses concise.\n\n{context}"
)

user_prompt="{user_input}"

def get_prompt_format():
    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{input}"},
    ])

    return prompt