from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import os 
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

memory = ConversationBufferMemory(
    memory_key="chat_history",  # Key used to store conversation
    return_messages=True        # Return as Message objects (needed for Chat models)
)


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)

def ask_groq_llm(prompt):
    # Load past conversation
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Prepare messages with memory
    messages = [
        SystemMessage(content="You are a helpful chatbot.")
    ] + chat_history + [
        HumanMessage(content=prompt)
    ]

    # Get response
    response = llm.invoke(messages)

    # Save new interaction to memory
    memory.save_context({"input": prompt}, {"output": response.content})

    return response.content


while True:
    prompt = input("Ask Groq: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    print(ask_groq_llm(prompt))
