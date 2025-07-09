#!/usr/bin/env python3
"""
Simplified Real-time Text-to-Speech with Configurable Parameters
"""

# ============================================================================
# UNIVERSAL CONSTANTS - MODIFY THESE TO TEST DIFFERENT SETTINGS
# ============================================================================

# Text to speak (modify this to test different phrases)
TEXT_TO_SPEAK = """
Hello, this is a test of the optimized text-to-speech system with natural human-like pacing.
"""

# Speaker ID (change this to test different voices)
# Examples: "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276", "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286", "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300", "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311", "p312", "p313", "p314", "p316", "p317", "p318", "p323", "p326", "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347", "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"
SPEAKER_ID = "p225"  # Female British accent

# Speed control (1.0 = normal, >1.0 = slower/more natural, <1.0 = faster)
SPEECH_STRETCH_FACTOR = 1.05

# Model configuration
MODEL_NAME = "tts_models/en/vctk/vits"
SAMPLE_RATE = 22050

# ============================================================================

import numpy as np
import subprocess
import tempfile
import os
import scipy.io.wavfile
import scipy.signal
import time
from TTS.api import TTS

try:
    import sounddevice as sd
except ImportError:
    sd = None

def try_stream_to_stdout(audio, sample_rate):
    """Stream audio directly to stdout pipe without temporary files"""
    try:
        print("🚀 Streaming via ffplay stdin...")
        
        # Convert audio to bytes (ensure proper format)
        audio_array = np.array(audio, dtype=np.float32)
        # Normalize to prevent clipping
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        
        # Try ffplay with stdin input
        ffplay_cmd = [
            'ffplay', '-f', 's16le', '-ar', str(sample_rate), 
            '-ac', '1', '-nodisp', '-autoexit', '-'
        ]
        
        process = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
        if process.stdin:
            process.stdin.write(audio_bytes)
            process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            print("✅ Direct streaming successful!")
            return True
        else:
            print(f"❌ Direct streaming failed with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Direct streaming failed: {e}")
        return False

def try_play_with_ffplay_file(audio, sample_rate):
    """Fallback: Use ffplay with temporary file"""
    try:
        print("🎵 Using ffplay with temporary file...")
        
        # Create temporary wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Ensure audio is properly formatted
        audio_array = np.array(audio, dtype=np.float32)
        
        # Save audio to temp file
        scipy.io.wavfile.write(temp_path, sample_rate, audio_array)
        
        # Play with ffplay
        result = subprocess.run(['ffplay', '-nodisp', '-autoexit', temp_path], 
                              capture_output=True, timeout=30, check=True)
        
        print("✅ Audio played successfully!")
        os.unlink(temp_path)  # Clean up temp file
        return True
        
    except Exception as e:
        print(f"❌ ffplay file method failed: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return False

def try_play_with_sounddevice(audio, sample_rate):
    """Try direct sounddevice playback"""
    try:
        if sd is None:
            print("❌ sounddevice not available")
            return False
            
        print("🔊 Using direct sounddevice...")
        
        # Ensure audio is numpy array and properly formatted
        audio_array = np.array(audio, dtype=np.float32)
        
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
        print("✅ Direct sounddevice playback successful!")
        return True
    except Exception as e:
        print(f"❌ Sounddevice playback failed: {e}")
        return False

def stream_audio(audio, sample_rate):
    """Stream audio using the best available method"""
    duration = len(audio) / sample_rate
    print(f"🎵 Audio ready: {len(audio)} samples at {sample_rate}Hz ({duration:.2f} seconds)")
    
    # Method 1: Direct stdout streaming (fastest, no temp files)
    if try_stream_to_stdout(audio, sample_rate):
        return
    
    # Method 2: Direct sounddevice (if available)
    if try_play_with_sounddevice(audio, sample_rate):
        return
    
    # Method 3: ffplay with temp file (most compatible)
    if try_play_with_ffplay_file(audio, sample_rate):
        return
    
    # Fallback: Save to file
    print("🔄 All streaming methods failed, saving to file...")
    scipy.io.wavfile.write("output.wav", sample_rate, np.array(audio, dtype=np.float32))
    print("💾 Audio saved to 'output.wav'")

def main():
    print("🚀 Configurable TTS System")
    print("=" * 60)
    print(f"📝 Text: '{TEXT_TO_SPEAK}'")
    print(f"🎭 Speaker: {SPEAKER_ID}")
    print(f"🐌 Speed Factor: {SPEECH_STRETCH_FACTOR}x")
    print("=" * 60)
    
    # Initialize TTS model
    print("📥 Loading TTS model...")
    tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
    print("✅ TTS model loaded successfully!")
    
    # Display available speakers for reference
    print(f"\n🎭 Available speakers: {len(tts.speakers)} total")
    print("   (Modify SPEAKER_ID constant to try different voices)")
    
    start_time = time.time()
    
    try:
        # Generate audio
        print(f"\n⚙️  Generating audio with speaker '{SPEAKER_ID}'...")
        audio = tts.tts(TEXT_TO_SPEAK, speaker=SPEAKER_ID)
        
        # Verify audio was generated
        if audio is None or len(audio) == 0:
            print("❌ Audio generation failed!")
            return
        
        generation_time = time.time() - start_time
        print(f"✅ Audio generated in {generation_time:.2f} seconds")
        
        # Apply speed adjustment for more natural speech
        if SPEECH_STRETCH_FACTOR != 1.0:
            print(f"🐌 Adjusting speech speed ({SPEECH_STRETCH_FACTOR}x)...")
            audio = scipy.signal.resample(audio, int(len(audio) * SPEECH_STRETCH_FACTOR))
        
        # Stream the audio
        print("🎵 Streaming audio...")
        stream_audio(audio, SAMPLE_RATE)
        
        total_time = time.time() - start_time
        audio_duration = len(audio) / SAMPLE_RATE
        real_time_factor = generation_time / audio_duration
        
        print(f"\n📊 Performance:")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Audio duration: {audio_duration:.2f}s")
        print(f"   Real-time factor: {real_time_factor:.2f}x")
        print(f"   Total time: {total_time:.2f}s")
        
        print("\n✅ TTS completed successfully!")
        print("\n💡 To customize:")
        print("   - Edit TEXT_TO_SPEAK to change what is spoken")
        print("   - Edit SPEAKER_ID to change voice (e.g., 'p225', 'p226', 'p227')")
        print("   - Edit SPEECH_STRETCH_FACTOR to adjust speed (>1.0 = slower)")
        
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        return

if __name__ == "__main__":
    main() 