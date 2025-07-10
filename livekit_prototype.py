#!/usr/bin/env python3
"""
LiveKit Voice Agent Prototype with Local STT/TTS Models
"""

import asyncio
import json
import tempfile
import os
import numpy as np
import scipy.io.wavfile
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

# LiveKit imports
from livekit import rtc

# Your existing model imports
from faster_whisper import WhisperModel
from TTS.api import TTS
import torch

@dataclass
class VoiceAgentConfig:
    """Configuration for the voice agent"""
    # LiveKit settings
    livekit_url: str = "ws://localhost:7880"
    room_name: str = "voice-agent-test"
    participant_name: str = "VoiceAgent"
    
    # STT settings
    stt_model_size: str = "small"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    
    # TTS settings
    tts_model_name: str = "tts_models/en/vctk/vits"
    tts_speaker_id: str = "p225"
    tts_sample_rate: int = 22050
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_channels: int = 1

class LocalSTT:
    """Local Speech-to-Text using faster-whisper"""
    
    def __init__(self, model_size: str, device: str = "cuda", compute_type: str = "float16"):
        print(f"🔧 Initializing STT model: {model_size} on {device}")
        
        # Set up CUDA environment (from your existing setup)
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu'
        else:
            os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:' + os.environ['LD_LIBRARY_PATH']
        
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"✅ STT model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load STT model on {device}: {e}")
            # Fallback to CPU
            print("🔄 Falling back to CPU...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("✅ STT model loaded on CPU")
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Convert bytes to numpy array and save
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            scipy.io.wavfile.write(temp_path, 16000, audio_array)  # Fixed sample rate
            
            # Transcribe
            segments, info = self.model.transcribe(temp_path, beam_size=5)
            segments_list = list(segments)
            
            # Clean up
            os.unlink(temp_path)
            
            if segments_list:
                text = segments_list[0].text.strip()
                print(f"🎤 STT: '{text}'")
                return text
            else:
                print("🎤 STT: No speech detected")
                return ""
                
        except Exception as e:
            print(f"❌ STT error: {e}")
            return ""

class LocalTTS:
    """Local Text-to-Speech using TTS library"""
    
    def __init__(self, model_name: str, speaker_id: str, sample_rate: int = 22050):
        print(f"🔧 Initializing TTS model: {model_name}")
        
        try:
            self.tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
            self.speaker_id = speaker_id
            self.sample_rate = sample_rate
            print(f"✅ TTS model loaded successfully")
            if hasattr(self.tts, 'speakers') and self.tts.speakers:
                print(f" Available speakers: {len(self.tts.speakers)}")
            print(f"🎭 Using speaker: {speaker_id}")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            raise
    
    def synthesize_speech(self, text: str) -> bytes:
        """Convert text to audio bytes"""
        try:
            print(f"🔊 TTS: '{text}'")
            
            # Generate audio
            audio = self.tts.tts(text, speaker=self.speaker_id)
            
            # Convert to bytes
            audio_array = np.array(audio, dtype=np.float32)
            # Normalize and convert to int16
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            
            print(f"✅ TTS: Generated {len(audio_bytes)} bytes of audio")
            return audio_bytes
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return b""

class LiveKitVoiceAgent:
    """Main voice agent class"""
    
    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self.room: Optional[rtc.Room] = None
        
        # Initialize models
        self.stt = LocalSTT(
            model_size=config.stt_model_size,
            device=config.stt_device,
            compute_type=config.stt_compute_type
        )
        
        self.tts = LocalTTS(
            model_name=config.tts_model_name,
            speaker_id=config.tts_speaker_id,
            sample_rate=config.tts_sample_rate
        )
        
        # Simple conversation history
        self.conversation_history = []
    
    def _generate_token(self) -> str:
        """Generate a JWT token for LiveKit authentication"""
        try:
            import jwt
            import time
            
            # Token payload
            payload = {
                "video": {"room": self.config.room_name, "roomJoin": True},
                "iss": "devkey",
                "sub": self.config.participant_name,
                "exp": int(time.time()) + 3600,  # 1 hour
                "iat": int(time.time())
            }
            
            # Generate token
            token = jwt.encode(payload, "secret", algorithm="HS256")
            return token
            
        except ImportError:
            print("⚠️  PyJWT not installed. Install with: pip install PyJWT")
            return "dev-token"
    
    async def connect_to_room(self):
        """Connect to LiveKit room"""
        try:
            print(f" Connecting to LiveKit room: {self.config.room_name}")
            
            # Create room
            self.room = rtc.Room()
            
            # Generate token
            token = self._generate_token()
            
            # Connect to room
            await self.room.connect(
                self.config.livekit_url,
                token=token
            )
            
            print(f"✅ Connected to room: {self.config.room_name}")
            print(f" Local participant: {self.room.local_participant.identity}")
            
            # Set up event handlers
            self.room.on("track_subscribed", self._on_track_subscribed)
            
            print("🎤 Voice agent ready to receive audio")
            
        except Exception as e:
            print(f"❌ Failed to connect to room: {e}")
            raise
    
    def _on_track_subscribed(self, track, publication, participant):
        """Handle incoming audio tracks"""
        print(f"🎧 Received track from: {participant.identity}")
        if hasattr(track, 'kind') and str(track.kind) == 'audio':
            print(f"🎵 Audio track received from {participant.identity}")
            # Set up audio data handler
            track.on("data_received", self._on_audio_data)
    
    def _on_audio_data(self, data: bytes):
        """Process incoming audio data"""
        try:
            print(f"🎵 Received {len(data)} bytes of audio data")
            
            # Transcribe audio
            text = self.stt.transcribe_audio(data)
            
            if text:
                # Add to conversation history
                self.conversation_history.append({"user": text, "timestamp": time.time()})
                
                # Generate response (simple echo for testing)
                response = self._generate_response(text)
                
                # Convert response to audio (for testing, save to file)
                audio_data = self.tts.synthesize_speech(response)
                
                if audio_data:
                    # Save audio response to file for testing
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    scipy.io.wavfile.write("agent_response.wav", self.config.tts_sample_rate, audio_array)
                    print("💾 Audio response saved to 'agent_response.wav'")
                    
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
    
    def _generate_response(self, user_text: str) -> str:
        """Generate a response to user input"""
        # Simple response logic - replace with your LLM
        responses = [
            f"I heard you say: {user_text}",
            f"That's interesting! You said: {user_text}",
            f"Let me think about: {user_text}",
            f"Thanks for sharing: {user_text}"
        ]
        
        # Simple round-robin response
        response_index = len(self.conversation_history) % len(responses)
        return responses[response_index]
    
    async def run(self):
        """Main run loop"""
        try:
            # Connect to room
            await self.connect_to_room()
            
            print("🎙️  Voice agent is running!")
            print("💡 Join the room from another client to test")
            print("📝 Agent will transcribe speech and save responses to 'agent_response.wav'")
            print("🛑 Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice agent...")
        except Exception as e:
            print(f"❌ Error in voice agent: {e}")
        finally:
            if self.room:
                await self.room.disconnect()
            print("👋 Voice agent stopped")

async def main():
    """Main function"""
    print("🚀 LiveKit Voice Agent Prototype")
    print("=" * 50)
    
    # Configuration
    config = VoiceAgentConfig(
        livekit_url="ws://localhost:7880",
        room_name="voice-agent-test",
        participant_name="VoiceAgent",
        stt_model_size="small",
        stt_device="cuda" if torch.cuda.is_available() else "cpu",
        stt_compute_type="float16",
        tts_model_name="tts_models/en/vctk/vits",
        tts_speaker_id="p225"
    )
    
    # Create and run voice agent
    agent = LiveKitVoiceAgent(config)
    await agent.run()

if __name__ == "__main__":
    # Check if LiveKit server is running
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 7880))
        sock.close()
        
        if result != 0:
            print("⚠️  LiveKit server not running on localhost:7880")
            print("💡 Start it with: docker-compose up -d")
            print(" Or run: docker-compose logs livekit")
            exit(1)
    except Exception as e:
        print(f"❌ Error checking LiveKit server: {e}")
        exit(1)
    
    # Run the voice agent
    asyncio.run(main()) 