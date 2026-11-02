"""This is what a prospect's engineer would actually write: two lines added
around code they already have. No knowledge of OpenTelemetry, no reading
mas_recon's source."""

import os

from mas_recon import start_capture

handle = start_capture()  # <-- line 1: add before building your agent(s)

from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402

# Swap this for however you already build your model -- Bedrock, OpenAI,
# Anthropic direct, a gateway. start_capture() doesn't care; it only watches
# the spans Strands already emits natively.
model = BedrockModel(model_id=os.environ.get("MODEL_ID", "global.anthropic.claude-sonnet-4-6"))
agent = Agent(name="demo_agent", model=model, system_prompt="You are a terse assistant. Answer in one sentence.")
result = agent("What is the capital of France?")
print("agent said:", result)

handle.stop()  # <-- line 2: after your agent run finishes
