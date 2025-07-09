from TTS.api import TTS
import sounddevice as sd
import numpy as np
import subprocess
import tempfile
import os
import scipy.io.wavfile

def try_stream_to_stdout(audio, sample_rate):
    """Stream audio directly to stdout pipe without temporary files"""
    try:
        print("🚀 Trying direct stdout streaming...")
        
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
        print("🎵 Trying ffplay with temporary file...")
        
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
        
        print("✅ Audio played successfully with ffplay!")
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
        print("🔊 Trying direct sounddevice playback...")
        
        # Ensure audio is numpy array and properly formatted
        audio_array = np.array(audio, dtype=np.float32)
        
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
        print("✅ Direct sounddevice playback successful!")
        return True
    except Exception as e:
        print(f"❌ Sounddevice playback failed: {e}")
        return False

def stream_audio_optimized(audio, sample_rate):
    """Optimized streaming with focus on consistency"""
    print(f"🎵 Audio ready: {len(audio)} samples at {sample_rate}Hz ({len(audio)/sample_rate:.2f} seconds)")
    
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
    print("🚀 Enhanced TTS with VCTK Multi-Speaker Model")
    print("=" * 60)
    
    # Initialize TTS model
    print("📥 Loading TTS model...")
    MODEL_NAME = "tts_models/en/vctk/vits"
    tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
    print("✅ TTS model loaded successfully!")
    
    # Display available speakers
    print("\n🎭 Available Speakers:")
    speakers = tts.speakers
    if speakers and len(speakers) > 0:
        print(f"Total speakers available: {len(speakers)}")
        
        # Show first 10 speakers as examples
        for i, speaker in enumerate(speakers[:10]):
            print(f"  {i+1:2d}. {speaker}")
        if len(speakers) > 10:
            print(f"  ... and {len(speakers)-10} more speakers")
        
        # Select a few interesting speakers for demonstration
        demo_speakers = [
            speakers[0],   # First speaker
            speakers[5] if len(speakers) > 5 else speakers[0],   # 6th speaker or first
            speakers[10] if len(speakers) > 10 else speakers[-1],  # 11th or last
        ]
        
        print(f"\n🎯 Demo will use speakers: {', '.join(demo_speakers)}")
    else:
        print("⚠️  No speakers found or single-speaker model")
        demo_speakers = None
    print("=" * 60)
    
    # Predefined text for consistent testing
    test_texts = [
        "Hello, this is a test of the optimized text-to-speech system.",
        "The audio quality should be consistent and clear.",
        "This version generates complete audio before streaming for better reliability.",
        "Streaming audio directly from Coqui TTS in real time.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    print("=" * 60)
    print("🗣️  Testing TTS with predefined texts...")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        # Select speaker for this test
        if demo_speakers:
            current_speaker = demo_speakers[(i-1) % len(demo_speakers)]
            print(f"\n🎤 Test {i}/{len(test_texts)} with speaker '{current_speaker}': '{text}'")
        else:
            current_speaker = None
            print(f"\n🎤 Test {i}/{len(test_texts)}: '{text}'")
        
        try:
            # Generate complete audio first (ensures consistency)
            print("⚙️  Generating audio...")
            if current_speaker:
                audio = tts.tts(text, speaker="p268")
            else:
                audio = tts.tts(text)
            
            # Post-process audio to make it slower and more natural
            # Stretch the audio to make it slower (more human-like)
            import scipy.signal
            stretch_factor = 1.15  # Make it 15% slower for more natural speech
            print(f"🐌 Slowing down speech by {stretch_factor}x for more natural pacing...")
            audio = scipy.signal.resample(audio, int(len(audio) * stretch_factor))
            sample_rate = 22050
            
            # Verify audio was generated properly
            if audio is None or len(audio) == 0:
                print("❌ Audio generation failed!")
                continue
                
            print(f"✅ Audio generated: {len(audio)} samples")
            
            # Now stream the complete audio
            print("🎵 Streaming audio...")
            stream_audio_optimized(audio, sample_rate)
            
            print(f"✅ Test {i} completed successfully!")
            
            # Small pause between tests
            import time
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🏁 All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
