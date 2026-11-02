"""Same two-line pattern for AutoGen (AG2 / autogen-agentchat)."""

import asyncio
import os

from mas_recon import start_capture

handle = start_capture()  # <-- line 1

from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination  # noqa: E402
from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402

model_client = OpenAIChatCompletionClient(
    model=os.environ.get("GATEWAY_MODEL_ID", "bedrock/global.anthropic.claude-sonnet-5"),
    base_url=os.environ.get("GATEWAY_BASE_URL", "https://litellm.prod.outshift.ai"),
    api_key=os.environ["OPENAI_API_KEY"],
    model_info={"vision": False, "function_calling": True, "json_output": True, "family": "unknown", "structured_output": False},
)

researcher = AssistantAgent(
    name="Researcher",
    model_client=model_client,
    system_message="You research facts. Give 2 bullet points about the topic, then say TERMINATE.",
)
writer = AssistantAgent(
    name="Writer",
    model_client=model_client,
    system_message="You write a one-sentence summary from the research notes above, then say TERMINATE.",
)

team = RoundRobinGroupChat([researcher, writer], termination_condition=MaxMessageTermination(4))


async def main():
    result = await team.run(task="Research 2 facts about rubber ducks, then summarize in one sentence.")
    print("result:", result.messages[-1].content)


asyncio.run(main())

handle.stop()  # <-- line 2
