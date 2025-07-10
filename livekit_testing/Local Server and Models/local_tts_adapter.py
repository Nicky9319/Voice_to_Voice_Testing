import asyncio
import tempfile
import os
import numpy as np
from typing import AsyncGenerator, Optional
from TTS.api import TTS
import scipy.io.wavfile
import scipy.signal

class LocalTTSAdapter:
    """Local TTS adapter using TTS library for LiveKit voice assistant"""
    
    def __init__(self, model_name: str = "tts_models/en/vctk/vits", speaker_id: str = "p225"):
        self.model_name = model_name
        self.speaker_id = speaker_id
        self.sample_rate = 22050
        self.tts_model = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the TTS model asynchronously"""
        if self._initialized:
            return
            
        # Run TTS model loading in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.tts_model = await loop.run_in_executor(
            None, 
            lambda: TTS(model_name=self.model_name, progress_bar=False, gpu=False)
        )
        self._initialized = True
        print(f"✅ Local TTS model loaded: {self.model_name}")
        
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio and yield audio chunks"""
        if not self._initialized:
            await self.initialize()
            
        # Generate audio in a thread pool
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: self.tts_model.tts(text, speaker=self.speaker_id)
        )
        
        if audio is None or len(audio) == 0:
            print("❌ Audio generation failed")
            return
            
        # Convert to proper format
        audio_array = np.array(audio, dtype=np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            
        # Convert to 16-bit PCM
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        
        # Yield audio in chunks (simulate streaming)
        chunk_size = 4096  # Adjust based on your needs
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.01)  # Small delay to simulate real-time streaming
            
    def get_sample_rate(self) -> int:
        """Get the sample rate of the generated audio"""
        return self.sample_rate 