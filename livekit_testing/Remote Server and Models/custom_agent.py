from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.agents.llm import ChatContext, ChatMessage, FunctionTool, RawFunctionTool

class StaticAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")
        self.current_user_message = None

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        self.current_user_message = new_message.content

        print("\n\n\n")
        print("========== CURRENT USER MESSAGE ==========")
        print(self.current_user_message)
        print("==========================================")
        print("\n\n\n")
        # You can add any additional processing here if needed
        await super().on_user_turn_completed(turn_ctx, new_message)