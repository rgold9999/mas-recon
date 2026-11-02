"""Same two-line pattern for the Claude Agent SDK, including subagent delegation.

The SDK shells out to the `claude` CLI as a child process -- it isn't a pure
Python client like the other examples. Make sure the CLI is installed and
authenticated before running this:

    npm install -g @anthropic-ai/claude-code   # or a local prefix if you lack
                                                # global npm permissions
    claude /login                              # interactive, one-time

If `claude` isn't on your PATH, set CLAUDE_CLI_PATH to its full location.
"""

import asyncio
import os

from mas_recon import start_capture

handle = start_capture()  # <-- line 1

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query  # noqa: E402

options = ClaudeAgentOptions(
    cli_path=os.environ.get("CLAUDE_CLI_PATH"),  # None = rely on PATH
    system_prompt=(
        "You coordinate a specialist to answer the user's request: delegate "
        "research to the 'researcher' subagent, then write a one-sentence "
        "summary yourself from what it returns."
    ),
    agents={
        "researcher": AgentDefinition(
            description="Researches facts about a topic and returns bullet points.",
            prompt="You research facts and return 2-3 bullet points of verifiable claims. Be concise.",
        ),
    },
    max_turns=6,
)


async def main():
    async for message in query(prompt="Give me 2 facts about the Eiffel Tower, then summarize in one sentence.", options=options):
        print(type(message).__name__, getattr(message, "content", getattr(message, "result", "")))


asyncio.run(main())

handle.stop()  # <-- line 2
