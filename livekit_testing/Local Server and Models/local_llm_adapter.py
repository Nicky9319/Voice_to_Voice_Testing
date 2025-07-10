import asyncio
import aiohttp
import json
from typing import AsyncGenerator, Optional, Dict, Any
from livekit.agents.llm import ChatContext, ChatMessage

class LocalLLMAdapter:
    """Local LLM adapter for LiveKit voice assistant"""
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = None
        
    async def initialize(self):
        """Initialize the HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def chat(self, chat_ctx: ChatContext) -> AsyncGenerator[str, None]:
        """Generate chat response using local LLM"""
        await self.initialize()
        
        # Convert LiveKit chat context to Ollama format
        messages = []
        for msg in chat_ctx.messages:
            if msg.role == "system":
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
                
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ LLM API error: {response.status} - {error_text}")
                    yield "I'm sorry, I'm having trouble processing your request right now."
                    return
                    
                # Stream the response
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'message' in data and 'content' in data['message']:
                                content = data['message']['content']
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"❌ Error parsing LLM response: {e}")
                            continue
                            
        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            yield "I'm sorry, I'm having trouble connecting to my language model right now."
            
    async def generate_response(self, prompt: str) -> str:
        """Generate a single response (non-streaming)"""
        # await self.initialize()
        
        # payload = {
        #     "model": self.model_name,
        #     "prompt": prompt,
        #     "stream": False,
        #     "options": {
        #         "temperature": 0.7,
        #         "top_p": 0.9,
        #         "max_tokens": 1000
        #     }
        # }
        
        # try:
        #     async with self.session.post(
        #         f"{self.base_url}/api/generate",
        #         json=payload,
        #         headers={"Content-Type": "application/json"}
        #     ) as response:
        #         if response.status != 200:
        #             error_text = await response.text()
        #             print(f"❌ LLM API error: {response.status} - {error_text}")
        #             return "I'm sorry, I'm having trouble processing your request right now."
                    
        #         data = await response.json()
        #         return data.get('response', '')
                
        # except Exception as e:
        #     print(f"❌ LLM request failed: {e}")
        #     return "I'm sorry, I'm having trouble connecting to my language model right now."

        return "I'm sorry, I'm having trouble connecting to my language model right now."

class MockLocalLLMAdapter:
    """Mock local LLM adapter for testing without Ollama"""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        
    async def chat(self, chat_ctx: ChatContext) -> AsyncGenerator[str, None]:
        """Generate mock chat response"""
        # Get the last user message
        user_message = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user":
                user_message = msg.content
                break
                
        # Simple response logic
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            yield "Hello! I'm your local AI assistant. How can I help you today?"
        elif "appointment" in user_message.lower() or "book" in user_message.lower():
            yield "I can help you book an appointment. Please provide your email and name."
        elif "help" in user_message.lower():
            yield "I'm here to help! I can assist with appointment booking and answer your questions."
        else:
            yield f"I understand you said: '{user_message}'. How can I assist you with that?"
            
    async def generate_response(self, prompt: str) -> str:
        """Generate a single mock response"""
        if "hello" in prompt.lower():
            return "Hello! I'm your local AI assistant."
        else:
            return f"I understand: {prompt}. How can I help?" 