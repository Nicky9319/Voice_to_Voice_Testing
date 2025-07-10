from flask import Flask, request, send_file, jsonify
import numpy as np
import tempfile
import os
import scipy.io.wavfile
from livekit_prototype import LocalSTT, LocalTTS, VoiceAgentConfig

app = Flask(__name__)

# Initialize models (reuse config from your prototype)
config = VoiceAgentConfig()
stt = LocalSTT(
    model_size=config.stt_model_size,
    device=config.stt_device,
    compute_type=config.stt_compute_type
)
tts = LocalTTS(
    model_name=config.tts_model_name,
    speaker_id=config.tts_speaker_id,
    sample_rate=config.tts_sample_rate
)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        file.save(temp_audio)
        temp_audio_path = temp_audio.name
    # Read audio bytes
    with open(temp_audio_path, 'rb') as f:
        audio_bytes = f.read()
    # Run STT
    transcript = stt.transcribe_audio(audio_bytes)
    # Generate response (echo)
    response_text = f"You said: {transcript}"
    # Run TTS
    tts_audio_bytes = tts.synthesize_speech(response_text)
    # Save TTS output to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
        audio_array = np.frombuffer(tts_audio_bytes, dtype=np.int16)
        scipy.io.wavfile.write(temp_out.name, config.tts_sample_rate, audio_array)
        out_path = temp_out.name
    # Clean up input temp file
    os.unlink(temp_audio_path)
    # Return the TTS audio file
    return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name='response.wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True) 