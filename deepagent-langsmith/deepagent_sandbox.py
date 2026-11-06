import os
from e2b import Sandbox
from langchain_e2b import E2BSandbox
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

e2b_sandbox = Sandbox.create()
sandbox_backend = E2BSandbox(sandbox=e2b_sandbox)

model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.211.1:11434",
)

agent = create_deep_agent(
    model=model,
    system_prompt="You are a mathematician.",
    backend=sandbox_backend,
)

try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Calculate 2+2 in Python"}]})
    print(result["messages"][-1].content)
finally:
    e2b_sandbox.kill()

# python3.11 deepagent_sandbox
# Here's a simple function in Python that calculates 2+2:

# ```
# def calculate(x, y):
#     return x + y

# result = calculate(2, 2)
# print(result)
# Output: 4

