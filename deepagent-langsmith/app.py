import os
import csv
import io
from e2b import Sandbox
from langchain_e2b import E2BSandbox
from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

slack_token = os.environ["SLACK_USER_TOKEN"]
slack_client = WebClient(token=slack_token)
channel = os.environ["CHANNEL_ID"]

e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)

@tool(parse_docstring=True)
def slack_send_message(text: str, file_path: str | None = None) -> str:
    """Send a message, optionally including attachments such as images.

    Args:
        text: text content of the message.
        file_path: file path of the attachment in the sandbox filesystem.
    """
    if not file_path:
        slack_client.chat_postMessage(channel=channel, text=text)
    else:
        fp = backend.download_files([file_path])
        slack_client.files_upload_v2(
            channel=channel,
            content=fp[0].content,
            initial_comment=text,
        )
    return "Message sent."

data = [
    ["Date", "Product", "Units Sold", "Revenue"],
    ["2025-08-01", "Widget A", 10, 250],
    ["2025-08-02", "Widget B", 5, 125],
    ["2025-08-03", "Widget A", 7, 175],
    ["2025-08-04", "Widget C", 3, 90],
    ["2025-08-05", "Widget B", 8, 200],
]

text_buf = io.StringIO()
writer = csv.writer(text_buf)
writer.writerows(data)
csv_bytes = text_buf.getvalue().encode("utf-8")
text_buf.close()

backend.upload_files([("data/sales_data.csv", csv_bytes)])

model = ChatOllama(
    model="qwen2.5:7b", 
    base_url="http://192.168.211.1:11434",
    num_ctx=8192,
    temperature=0.0
)

agent = create_deep_agent(
    model=model,
    tools=[slack_send_message],
    backend=backend,
    checkpointer=InMemorySaver()
)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

prompt_content = (
    "Analyze data/sales_data.csv in the current dir and generate a beautiful plot. "
    "When finished, send your analysis and the plot file path to Slack using the tool."
)
input_payload = {"messages": [HumanMessage(content=prompt_content)]}

try:
    for event in agent.stream(input_payload, config=config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            latest_msg = event["messages"][-1]
            latest_msg.pretty_print()

finally:
    e2b_sandbox.kill()