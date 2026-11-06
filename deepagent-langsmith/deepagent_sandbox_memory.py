import os
from e2b import Sandbox
from langchain_e2b import E2BSandbox
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

e2b_sandbox = Sandbox.create()
sandbox_backend = E2BSandbox(sandbox=e2b_sandbox)

# 1. Force the model to be strict and drop creative parameters
model = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://192.168.211.1:11434",
    num_ctx=8192,
    temperature=0.0  # CRITICAL: 0 stops the model from imagining 'make build' errors
)

# 2. Tell the framework to block extra agent middleware tools
agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a strict conversational QnA assistant. "
        "Do NOT try to launch subagents. Do NOT use shell tools or make commands. "
        "Read the conversation transcript history. Your only task is to remember the human's details and answer their text questions directly."
    ),
    backend=sandbox_backend,
    checkpointer=InMemorySaver(),
    # If the deepagents framework allows tool overrides, pass an empty list:
    # tools=[] 
)

try:
    thread_id = "test-thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤖 Telling the agent your name...")
    result = agent.invoke(
        {"messages": [HumanMessage(content="My name is Alice.")]}, 
        config=config
    )

    print("\nNow asking the agent: What is my name?")

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is my name?")]}, 
        config=config
    )

    print("\n--- Diagnostic State History ---")
    for msg in result["messages"]:
        # Only print if there is actual message content to clean up logs
        if msg.content:
            print(f"{msg.type}: {msg.content}")
    print("--------------------------------\n")

    print("✨ Agent Response:")
    print(result["messages"][-1].content)
finally:
    e2b_sandbox.kill()

