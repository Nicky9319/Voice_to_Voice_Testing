import asyncio
from typing import Annotated
import re
import os
import requests
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant

# Import our local adapters
from local_tts_adapter import LocalTTSAdapter
from local_stt_adapter import LocalSTTAdapter
from local_llm_adapter import LocalLLMAdapter, MockLocalLLMAdapter
from local_vad_adapter import LocalVADAdapter, SimpleVADAdapter

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env")

class AssistantFunction(agents.llm.FunctionContext):
    """This class defines functions that the assistant will call."""

    @agents.llm.ai_callable(
        description="Called when a user wants to book an appointment. This function sends a booking link to the provided email address and name."
    )
    async def book_appointment(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The email address to send the booking link to"
            ),
        ],
        name: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The name of the person booking the appointment"
            ),
        ],
    ):
        # Validate email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "The email address seems incorrect. Please provide a valid one."

        # Webhook call to book the appointment
        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            headers = {'Content-Type': 'application/json'}
            data = {'email': email, 'name': name}
            response = requests.post(webhook_url, json=data, headers=headers)
            response.raise_for_status()

            # Return success message
            return f"Appointment booking link sent to {email}. Please check your email."

        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your appointment. Please try again later."

    async def check_appointment_status(
        self,
        email: str,
    ):
        """Check if a user has booked an appointment based on their email."""
        api_token = os.getenv('API_TOKEN')
        print("calling check function")

        try:
            api_url = f"{os.getenv('CRM_CONTACT_LOOKUP_ENDPOINT')}?email={email}"
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            # Check if the contact has the 'livekit_appointment_booked' tag
            for contact in data.get('contacts', []):
                if 'livekit_appointment_booked' in contact.get('tags', []):
                    return "The user has successfully booked the appointment."
            return "The user has not yet booked an appointment. Please offer him help"

        except requests.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error checking the appointment status."

class LocalVoiceAssistant:
    """Local voice assistant using local TTS, STT, LLM, and VAD"""
    
    def __init__(self, 
                 tts_adapter: LocalTTSAdapter,
                 stt_adapter: LocalSTTAdapter,
                 llm_adapter: LocalLLMAdapter,
                 vad_adapter: LocalVADAdapter,
                 chat_context: ChatContext,
                 fnc_ctx: AssistantFunction):
        self.tts_adapter = tts_adapter
        self.stt_adapter = stt_adapter
        self.llm_adapter = llm_adapter
        self.vad_adapter = vad_adapter
        self.chat_context = chat_context
        self.fnc_ctx = fnc_ctx
        self.is_speaking = False
        
    async def initialize(self):
        """Initialize all adapters"""
        print("🚀 Initializing local voice assistant...")
        
        # Initialize all adapters
        await self.tts_adapter.initialize()
        await self.stt_adapter.initialize()
        await self.llm_adapter.initialize()
        await self.vad_adapter.initialize()
        
        print("✅ Local voice assistant initialized!")
        
    async def say(self, text: str, allow_interruptions: bool = True):
        """Speak the given text"""
        if self.is_speaking and not allow_interruptions:
            return
            
        self.is_speaking = True
        print(f"🗣️  Speaking: {text}")
        
        try:
            async for audio_chunk in self.tts_adapter.synthesize(text):
                # Here you would send the audio chunk to the WebRTC stream
                # For now, we'll just simulate it
                await asyncio.sleep(0.01)  # Simulate audio playback
                
        except Exception as e:
            print(f"❌ TTS error: {e}")
        finally:
            self.is_speaking = False
            
    async def process_speech(self, audio_data: bytes):
        """Process speech audio and generate response"""
        print("🎤 Processing speech...")
        
        # Transcribe audio
        transcription = await self.stt_adapter.transcribe_audio_data(audio_data)
        if not transcription:
            print("❌ Failed to transcribe audio")
            return
            
        print(f"📝 Transcription: {transcription}")
        
        # Add to chat context
        self.chat_context.messages.append(ChatMessage(role="user", content=transcription))
        
        # Generate response
        response_text = ""
        async for chunk in self.llm_adapter.chat(self.chat_context):
            response_text += chunk
            
        # Add assistant response to context
        self.chat_context.messages.append(ChatMessage(role="assistant", content=response_text))
        
        print(f"🤖 Response: {response_text}")
        
        # Speak the response
        await self.say(response_text)
        
    async def start_voice_detection(self):
        """Start voice activity detection"""
        async def on_speech_start():
            print("🎤 Speech started")
            
        async def on_speech_end(audio_data):
            print("🔇 Speech ended, processing...")
            await self.process_speech(audio_data)
            
        await self.vad_adapter.start_voice_detection(on_speech_start, on_speech_end)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Daela, a sales assistant for Knolabs AI Agency. "
                    "You offer appointment booking for AI/Automation services through voice interaction."
                ),
            )
        ]
    )

    # Initialize local adapters
    print("🔧 Setting up local services...")
    
    # TTS adapter
    tts_adapter = LocalTTSAdapter(
        model_name="tts_models/en/vctk/vits",
        speaker_id="p225"  # Female British accent
    )
    
    # STT adapter
    stt_adapter = LocalSTTAdapter(
        model_size="small",
        device="auto"  # Will try GPU first, fallback to CPU
    )
    
    # LLM adapter - use mock for testing, or LocalLLMAdapter for Ollama
    llm_adapter = MockLocalLLMAdapter(model_name="mock-model")
    # Uncomment below to use Ollama:
    # llm_adapter = LocalLLMAdapter(model_name="llama3.2", base_url="http://localhost:11434")
    
    # VAD adapter - use simple energy-based detection
    vad_adapter = SimpleVADAdapter(threshold=0.01)
    # Uncomment below to use webrtcvad:
    # vad_adapter = LocalVADAdapter(aggressiveness=2)
    
    # Create local voice assistant
    assistant = LocalVoiceAssistant(
        tts_adapter=tts_adapter,
        stt_adapter=stt_adapter,
        llm_adapter=llm_adapter,
        vad_adapter=vad_adapter,
        chat_context=chat_context,
        fnc_ctx=AssistantFunction()
    )
    
    # Initialize the assistant
    await assistant.initialize()
    
    # Start voice detection in background
    voice_task = asyncio.create_task(assistant.start_voice_detection())
    
    # Send initial greeting
    await asyncio.sleep(2)  # Wait for initialization
    await assistant.say("Hi there! I'm your local AI assistant. How can I help?", allow_interruptions=True)

    # Keep the connection alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)
        
    # Cleanup
    voice_task.cancel()
    try:
        await voice_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint)) 