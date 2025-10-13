import os
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant. Answer all questions to the best of your ability in one or two sentences."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chat_history.append(HumanMessage(content=user_input)) # Append user's message
    ai_msg = chain.invoke({"messages": chat_history}) # Invoke the chain
    print(ai_msg)
    chat_history.append(AIMessage(content=ai_msg.content)) # Append AI response to history
    print(f"AI: {ai_msg.content}") # Print AI response
