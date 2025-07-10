import asyncio
import numpy as np
import webrtcvad
import pyaudio
from typing import Optional, Callable
import threading
import queue

class LocalVADAdapter:
    """Local VAD adapter using webrtcvad for LiveKit voice assistant"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.vad = None
        self._initialized = False
        
        # Audio settings
        self.chunk_size = 480  # 30ms at 16kHz
        self.channels = 1
        self.format = pyaudio.paInt16
        
    async def initialize(self):
        """Initialize the VAD model"""
        if self._initialized:
            return
            
        try:
            self.vad = webrtcvad.Vad(self.aggressiveness)
            self._initialized = True
            print(f"✅ Local VAD initialized with aggressiveness {self.aggressiveness}")
        except Exception as e:
            print(f"❌ Failed to initialize VAD: {e}")
            raise
            
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech"""
        if not self._initialized:
            return True  # Default to True if not initialized
            
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return True  # Default to True on error
            
    async def start_voice_detection(self, on_speech_start: Callable, on_speech_end: Callable):
        """Start real-time voice activity detection"""
        if not self._initialized:
            await self.initialize()
            
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("🎤 Voice activity detection started. Press Ctrl+C to stop.")
            
            speech_buffer = []
            is_speaking = False
            silence_frames = 0
            max_silence_frames = 10  # 300ms of silence to end speech
            
            while True:
                try:
                    # Read audio data
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Check if this chunk contains speech
                    if self.is_speech(data):
                        if not is_speaking:
                            # Speech started
                            is_speaking = True
                            silence_frames = 0
                            await on_speech_start()
                            print("🎤 Speech detected")
                            
                        speech_buffer.append(data)
                        silence_frames = 0
                    else:
                        if is_speaking:
                            silence_frames += 1
                            
                            # If we've had enough silence, end speech
                            if silence_frames >= max_silence_frames:
                                is_speaking = False
                                await on_speech_end(b''.join(speech_buffer))
                                speech_buffer = []
                                print("🔇 Speech ended")
                                
                except KeyboardInterrupt:
                    break
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    def get_audio_format(self) -> tuple:
        """Get audio format (channels, sample_width, sample_rate)"""
        return (self.channels, 2, self.sample_rate)  # 2 bytes for Int16

class SimpleVADAdapter:
    """Simple VAD adapter using energy-based detection"""
    
    def __init__(self, threshold: float = 0.01, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = 480  # 30ms at 16kHz
        self.channels = 1
        self.format = pyaudio.paInt16
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech using energy threshold"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize energy (assuming 16-bit audio)
            normalized_energy = energy / 32768.0
            
            return normalized_energy > self.threshold
            
        except Exception as e:
            print(f"❌ Energy-based VAD error: {e}")
            return True  # Default to True on error
            
    async def start_voice_detection(self, on_speech_start: Callable, on_speech_end: Callable):
        """Start real-time voice activity detection using energy threshold"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("🎤 Energy-based voice detection started. Press Ctrl+C to stop.")
            
            speech_buffer = []
            is_speaking = False
            silence_frames = 0
            max_silence_frames = 10  # 300ms of silence to end speech
            
            while True:
                try:
                    # Read audio data
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Check if this chunk contains speech
                    if self.is_speech(data):
                        if not is_speaking:
                            # Speech started
                            is_speaking = True
                            silence_frames = 0
                            await on_speech_start()
                            print("🎤 Speech detected (energy-based)")
                            
                        speech_buffer.append(data)
                        silence_frames = 0
                    else:
                        if is_speaking:
                            silence_frames += 1
                            
                            # If we've had enough silence, end speech
                            if silence_frames >= max_silence_frames:
                                is_speaking = False
                                await on_speech_end(b''.join(speech_buffer))
                                speech_buffer = []
                                print("🔇 Speech ended (energy-based)")
                                
                except KeyboardInterrupt:
                    break
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    def get_audio_format(self) -> tuple:
        """Get audio format (channels, sample_width, sample_rate)"""
        return (self.channels, 2, self.sample_rate)  # 2 bytes for Int16 