import asyncio
import tempfile
import os
import numpy as np
from typing import Optional, Tuple
from faster_whisper import WhisperModel
import torch
import wave
import pyaudio

class LocalSTTAdapter:
    """Local STT adapter using faster-whisper for LiveKit voice assistant"""
    
    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self._initialized = False
        
        # Audio recording settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
    async def initialize(self):
        """Initialize the STT model asynchronously"""
        if self._initialized:
            return
            
        # Set LD_LIBRARY_PATH for cuDNN compatibility
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu'
        else:
            os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:' + os.environ['LD_LIBRARY_PATH']
            
        # Run model loading in a thread pool
        loop = asyncio.get_event_loop()
        
        try:
            # Try GPU first
            if torch.cuda.is_available() and self.device in ["auto", "cuda"]:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(self.model_size, device="cuda", compute_type="float16")
                )
                print(f"✅ Local STT model loaded on GPU: {self.model_size}")
            else:
                # Fallback to CPU
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(self.model_size, device="cpu", compute_type="int8")
                )
                print(f"✅ Local STT model loaded on CPU: {self.model_size}")
                
            self._initialized = True
            
        except Exception as e:
            print(f"❌ Failed to load STT model: {e}")
            # Try CPU as fallback
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(self.model_size, device="cpu", compute_type="int8")
                )
                print(f"✅ Local STT model loaded on CPU (fallback): {self.model_size}")
                self._initialized = True
            except Exception as e2:
                print(f"❌ Failed to load STT model on CPU: {e2}")
                raise
                
    async def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """Transcribe an audio file"""
        if not self._initialized:
            await self.initialize()
            
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
            
        try:
            # Run transcription in a thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(audio_file_path, beam_size=5)
            )
            
            # Convert generator to list and extract text
            segments_list = list(segments)
            text = " ".join([segment.text.strip() for segment in segments_list])
            
            print(f"✅ Transcription completed: '{text}'")
            return text
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
            
    async def transcribe_audio_data(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data from memory"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
                
            # Transcribe the temporary file
            result = await self.transcribe_audio_file(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Audio data transcription failed: {e}")
            return None
            
    async def start_realtime_transcription(self, callback):
        """Start real-time transcription from microphone"""
        if not self._initialized:
            await self.initialize()
            
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print("🎤 Real-time transcription started. Press Ctrl+C to stop.")
            
            while True:
                try:
                    # Read audio data
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    
                    # Process audio data (you might want to accumulate data for better results)
                    # For now, we'll just pass it to the callback
                    await callback(data)
                    
                except KeyboardInterrupt:
                    break
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    def get_audio_format(self) -> Tuple[int, int, int]:
        """Get audio format (channels, sample_width, sample_rate)"""
        return (self.channels, 2, self.rate)  # 2 bytes for Int16 