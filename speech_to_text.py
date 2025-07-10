import os
import sys
import subprocess
from faster_whisper import WhisperModel
import torch

# Set LD_LIBRARY_PATH to include cuDNN libraries
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu'
else:
    os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:' + os.environ['LD_LIBRARY_PATH']

def check_system_info():
    """Check system information for debugging"""
    print("=== SYSTEM DIAGNOSTICS ===")
    
    # Check CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') else 'Unknown'}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Check environment variables
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Check for cuDNN libraries
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
        cudnn_libs = [line for line in result.stdout.split('\n') if 'cudnn' in line.lower()]
        print(f"cuDNN libraries found: {len(cudnn_libs)}")
        for lib in cudnn_libs[:5]:  # Show first 5
            print(f"  {lib}")
    except Exception as e:
        print(f"Error checking cuDNN libraries: {e}")
    
    print("=" * 30)

def try_gpu_model(model_size):
    """Try to create GPU model with detailed error handling"""
    print(f"\n=== ATTEMPTING GPU MODEL CREATION ===")
    print(f"Model size: {model_size}")
    print(f"Device: cuda")
    
    # Try different compute types that work with cuDNN 8.x
    compute_types = [
        ("float16", "float16"),  # Try float16 first now that we have cuDNN 9.x
        ("int8_float16", "int8_float16"),
        ("int8", "int8"),
    ]
    
    try:
        # Test basic CUDA operations first
        print("Testing basic CUDA operations...")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print("✓ Basic CUDA operations work")
        else:
            raise RuntimeError("CUDA not available")
        
        # Try different compute types
        for compute_name, compute_type in compute_types:
            print(f"Trying compute type: {compute_name}...")
            try:
                model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                print(f"✓ GPU model created successfully with {compute_name}")
                return model, f"gpu_{compute_name}"
            except Exception as e:
                print(f"✗ Failed with {compute_name}: {e}")
                if "cuDNN" in str(e) or "cudnn" in str(e).lower():
                    print(f"  (cuDNN compatibility issue with {compute_name})")
                continue
        
        # If all GPU attempts failed
        print("✗ All GPU compute types failed")
        return None, "gpu_all_failed"
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return None, "import_error"
    except RuntimeError as e:
        print(f"✗ Runtime error: {e}")
        if "cuDNN" in str(e) or "cudnn" in str(e).lower():
            return None, "cudnn_error"
        return None, "runtime_error"
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        return None, "unknown_error"

def try_cpu_model(model_size):
    """Create CPU model as fallback"""
    print(f"\n=== FALLING BACK TO CPU ===")
    print(f"Model size: {model_size}")
    print(f"Device: cpu")
    print(f"Compute type: int8")
    
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("✓ CPU model created successfully")
        return model, "cpu"
    except Exception as e:
        print(f"✗ CPU model creation failed: {e}")
        return None, "cpu_error"

def transcribe_audio(model, audio_file):
    """Transcribe audio with error handling"""
    print(f"\n=== TRANSCRIBING AUDIO ===")
    print(f"Audio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"✗ Audio file not found: {audio_file}")
        return None, None
    
    try:
        # Transcribe with beam search
        segments, info = model.transcribe(audio_file, beam_size=5)
        
        # Convert generator to list to avoid issues
        segments_list = list(segments)
        
        print(f"✓ Transcription completed")
        print(f"Detected language: '{info.language}' with probability {info.language_probability:.4f}")
        print(f"Number of segments: {len(segments_list)}")
        
        return segments_list, info
        
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return None, None

def main():
    """Main function with comprehensive error handling"""
    print("=== FASTER WHISPER SPEECH-TO-TEXT ===")
    
    # System diagnostics
    check_system_info()
    
    # Model configuration
    model_size = "small"
    audio_file = "output.wav"
    
    # Force GPU first - don't fallback to CPU immediately
    print("\n=== FORCING GPU USAGE ===")
    model, model_type = try_gpu_model(model_size)
    
    # Only fallback to CPU if GPU completely fails
    if model is None:
        print("\n=== GPU FAILED, FALLING BACK TO CPU ===")
        model, model_type = try_cpu_model(model_size)
    
    # If both fail, exit
    if model is None:
        print("\n✗ Failed to create model with both GPU and CPU")
        print("Please check your installation and try again")
        sys.exit(1)
    
    print(f"\n✓ Using {model_type.upper()} model")
    
    # Transcribe audio
    segments, info = transcribe_audio(model, audio_file)
    
    if segments is None:
        print("\n✗ Transcription failed")
        sys.exit(1)
    
    # Print results
    print(f"\n=== TRANSCRIPTION RESULTS ===")
    with open("transcription.txt", "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            print(line)
            f.write(line + "\n")
    print(f"\n✓ Transcription completed successfully using {model_type.upper()}")
    print("Transcription saved to transcription.txt")

if __name__ == "__main__":
    main()
