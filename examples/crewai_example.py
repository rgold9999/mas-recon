"""Same two-line pattern for CrewAI."""

import os

from mas_recon import start_capture

handle = start_capture()  # <-- line 1

from crewai import LLM, Agent, Crew, Task  # noqa: E402

llm = LLM(
    model="openai/" + os.environ.get("GATEWAY_MODEL_ID", "bedrock/global.anthropic.claude-sonnet-5"),
    base_url=os.environ.get("GATEWAY_BASE_URL", "https://litellm.prod.outshift.ai"),
)

researcher = Agent(
    role="Researcher",
    goal="Find 2 facts about the given topic",
    backstory="You are a meticulous researcher.",
    llm=llm,
)
writer = Agent(
    role="Writer",
    goal="Write a one-sentence summary from research notes",
    backstory="You are a concise technical writer.",
    llm=llm,
)

research_task = Task(
    description="Research 2 facts about rubber ducks.",
    expected_output="2 bullet points of facts.",
    agent=researcher,
)
write_task = Task(
    description="Write a one-sentence summary from the research notes.",
    expected_output="One sentence.",
    agent=writer,
    context=[research_task],
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
print("result:", result)

handle.stop()  # <-- line 2
