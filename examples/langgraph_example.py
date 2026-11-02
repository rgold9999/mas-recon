"""Same two-line pattern for a LangGraph app. Agent names are auto-detected
from the real execution -- "Comedian" and "Critic" below are never told to
mas_recon anywhere; they're read straight off your own graph."""

import os

from mas_recon import start_capture

handle = start_capture()  # <-- line 1

from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

llm = ChatOpenAI(
    model=os.environ.get("GATEWAY_MODEL_ID", "bedrock/global.anthropic.claude-sonnet-5"),
    base_url=os.environ.get("GATEWAY_BASE_URL", "https://litellm.prod.outshift.ai"),
)


class State(TypedDict):
    topic: str
    joke: str
    rating: str


def comedian(state: State) -> dict:
    resp = llm.invoke(f"Write a one-sentence joke about {state['topic']}.")
    return {"joke": resp.content}


def critic(state: State) -> dict:
    resp = llm.invoke(f"Rate this joke 1-10, just the number: {state['joke']}")
    return {"rating": resp.content}


graph = StateGraph(State)
graph.add_node("Comedian", comedian)
graph.add_node("Critic", critic)
graph.set_entry_point("Comedian")
graph.add_edge("Comedian", "Critic")
graph.add_edge("Critic", END)
app = graph.compile()

result = app.invoke({"topic": "rubber ducks"})
print("result:", result)

handle.stop()  # <-- line 2
