from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.agents.llm import ChatContext, FunctionTool, RawFunctionTool
from collections.abc import AsyncIterable, AsyncGenerator

class StaticAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")
    
    def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[str, None]:
        async def gen():
            yield "this is a static hardcoded response"
        return gen()