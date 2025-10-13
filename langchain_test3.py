import os
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)

workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. Answer all questions to the best of your ability in one or two sentences."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

while True:
    # user_input = input("You: ")
    # if user_input.lower() in ["exit", "quit"]:
    #     break
    # chat_history.append(HumanMessage(content=user_input)) # Append user's message
    # ai_msg = chain.invoke({"messages": chat_history}) # Invoke the chain
    # chat_history.append(AIMessage(content=ai_msg.content)) # Append AI response to history
    # print(f"AI: {ai_msg.content}") # Print AI response
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    app.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": "1"}},)