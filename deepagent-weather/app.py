from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.211.1:11434",
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
    
agent = create_deep_agent(
    #model="ollama:llama3.2:3b",
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

final_message = res["messages"][-1]
print(final_message.content)
