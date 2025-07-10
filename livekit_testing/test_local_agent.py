import asyncio
import os
from pathlib import Path
import numpy as np

# Import adapters from parent directory
import sys
sys.path.append('..')
from local_tts_adapter import LocalTTSAdapter
from local_stt_adapter import LocalSTTAdapter
import subprocess
import soundfile as sf

OUTPUT_WAV = "output.wav"
TRANSCRIPT_TXT = "transcript.txt"

async def test_tts(text: str, wav_path: str):
    print(f"[TTS] Synthesizing speech for: '{text}' -> {wav_path}")
    tts = LocalTTSAdapter()
    await tts.initialize()
    audio_chunks = []
    async for chunk in tts.synthesize(text):
        audio_chunks.append(chunk)
    audio_bytes = b''.join(audio_chunks)
    # Convert bytes to int16 numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    # Save as WAV using soundfile (assume mono, 22050Hz)
    sf.write(wav_path, audio_array, tts.sample_rate, subtype='PCM_16')
    print(f"[TTS] Audio saved to {os.path.abspath(wav_path)}")

async def test_stt(wav_path: str, transcript_path: str):
    print(f"[STT] Transcribing audio from: {wav_path}")
    stt = LocalSTTAdapter()
    await stt.initialize()
    if not os.path.exists(wav_path):
        print(f"[STT] File not found: {wav_path}")
        return
    text = await stt.transcribe_audio_file(wav_path)
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(text or "")
    print(f"[STT] Transcript saved to {transcript_path}")

def is_valid_wav(filepath):
    try:
        with sf.SoundFile(filepath) as f:
            return f.format == 'WAV'
    except Exception:
        return False

async def main():
    # 1. TTS test
    test_phrase = "Hello, this is a test of the local TTS system."
    await test_tts(test_phrase, OUTPUT_WAV)

    # Check if the file is a valid WAV, if not, try to convert it
    if not is_valid_wav(OUTPUT_WAV):
        print(f"[WARN] {OUTPUT_WAV} is not a valid WAV file. Attempting conversion with ffmpeg...")
        output_fixed = "output_fixed.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", OUTPUT_WAV, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_fixed
            ], check=True)
            print(f"[INFO] Converted to {output_fixed}")
            wav_for_stt = output_fixed
        except Exception as e:
            print(f"[ERROR] ffmpeg conversion failed: {e}")
            wav_for_stt = OUTPUT_WAV
    else:
        wav_for_stt = OUTPUT_WAV

    # 2. STT test
    await test_stt(wav_for_stt, TRANSCRIPT_TXT)
    # 3. Show results
    print("\n[RESULTS]")
    print(f"Audio file: {OUTPUT_WAV}")
    if os.path.exists(TRANSCRIPT_TXT):
        with open(TRANSCRIPT_TXT, 'r', encoding='utf-8') as f:
            print(f"Transcript: {f.read().strip()}")
    else:
        print("Transcript file not found.")

if __name__ == "__main__":
    asyncio.run(main()) 