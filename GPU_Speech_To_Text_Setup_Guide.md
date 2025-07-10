# GPU-Accelerated Speech-to-Text Setup Guide for WSL2

## Overview

This guide documents the complete setup process for running GPU-accelerated speech-to-text using faster-whisper in WSL2 (Windows Subsystem for Linux 2). The setup enables real-time transcription using NVIDIA GPU acceleration.

## Prerequisites

- Windows 10/11 with WSL2 enabled
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on Windows
- WSL2 with Ubuntu 20.04 or later
- Conda/Miniconda installed in WSL2

## Technical Background

### The Problem
The main challenge was getting faster-whisper to use GPU acceleration in WSL2. The issues were:

1. **cuDNN Version Mismatch**: PyTorch expects specific cuDNN versions (9.1.0) but system had different versions (9.10.2)
2. **Library Path Issues**: Dynamic linker couldn't find cuDNN libraries due to missing environment variables
3. **Binary Compatibility**: Different cuDNN builds weren't compatible with PyTorch's expectations

### The Solution
- Install compatible PyTorch with CUDA support
- Install cuDNN libraries and create proper symlinks
- Configure system-wide library paths
- Set environment variables for dynamic linking

## Step-by-Step Setup

### 1. Verify GPU Availability

```bash
# Check if GPU is accessible from WSL
nvidia-smi
```

Expected output should show your NVIDIA GPU with CUDA version.

### 2. Install CUDA Toolkit and Libraries

```bash
# Update package lists
sudo apt update

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-0
```

### 3. Install cuDNN Libraries

```bash
# Install cuDNN 8.x runtime libraries
sudo apt install libcudnn8

# Install cuDNN 9.x for newer PyTorch compatibility
sudo apt install libcudnn9-cuda-12
```

### 4. Create Conda Environment

```bash
# Create new conda environment
conda create -n voice_to_voice python=3.10
conda activate voice_to_voice

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install faster-whisper
pip install faster-whisper
```

### 5. Create cuDNN Symlinks

The critical step: Create versioned symlinks that PyTorch expects.

```bash
cd /lib/x86_64-linux-gnu

# Remove any existing problematic symlinks
sudo rm -f libcudnn_ops.so.9.1.0 libcudnn_ops.so.9.1
sudo rm -f libcudnn_cnn.so.9.1.0 libcudnn_cnn.so.9.1
sudo rm -f libcudnn_adv.so.9.1.0 libcudnn_adv.so.9.1

# Create symlinks pointing to actual library files
sudo ln -sf libcudnn_ops.so.9.10.2 libcudnn_ops.so.9.1.0
sudo ln -sf libcudnn_ops.so.9.10.2 libcudnn_ops.so.9.1
sudo ln -sf libcudnn_cnn.so.9.10.2 libcudnn_cnn.so.9.1.0
sudo ln -sf libcudnn_cnn.so.9.10.2 libcudnn_cnn.so.9.1
sudo ln -sf libcudnn_adv.so.9.10.2 libcudnn_adv.so.9.1.0
sudo ln -sf libcudnn_adv.so.9.10.2 libcudnn_adv.so.9.1

# Update library cache
sudo ldconfig
```

### 6. Configure System-Wide Library Paths

```bash
# Create system-wide library configuration
echo "/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/cudnn.conf

# Update library cache
sudo ldconfig
```

### 7. Set User Environment Variables

```bash
# Add to bash profile for permanent setting
echo 'export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Reload profile
source ~/.bashrc
```

### 8. Verify Installation

```bash
# Check cuDNN libraries
ldconfig -p | grep cudnn

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Speech-to-Text Script

### Complete Script with Error Handling

```python
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
```

## Usage

### Basic Usage
```bash
# Activate environment
conda activate voice_to_voice

# Run speech-to-text
python speech_to_text.py
```

### Expected Output
```
=== FASTER WHISPER SPEECH-TO-TEXT ===
=== SYSTEM DIAGNOSTICS ===
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 90100
GPU count: 1
GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU
  Memory: 4.00 GB
  Compute capability: 8.6
LD_LIBRARY_PATH: /lib/x86_64-linux-gnu
cuDNN libraries found: 18
==============================

=== FORCING GPU USAGE ===
✓ Using GPU_FLOAT16 model

=== TRANSCRIBING AUDIO ===
✓ Transcription completed
Detected language: 'en' with probability 0.9932

=== TRANSCRIPTION RESULTS ===
[0.00s -> 3.60s] streaming audio directly from CokeWits in real time.

✓ Transcription completed successfully using GPU_FLOAT16
Transcription saved to transcription.txt
```

## Troubleshooting

### Common Issues

1. **"Unable to load cuDNN libraries"**
   - Check if symlinks are created correctly: `ls -la /lib/x86_64-linux-gnu/libcudnn*`
   - Verify library cache: `sudo ldconfig`

2. **"CUDA not available"**
   - Check GPU: `nvidia-smi`
   - Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

3. **"cuDNN version mismatch"**
   - Reinstall PyTorch with correct CUDA version
   - Create proper symlinks as shown in step 5

### Verification Commands

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check cuDNN libraries
ldconfig -p | grep cudnn

# Check environment
echo $LD_LIBRARY_PATH
```

## Performance Comparison

- **CPU-only**: ~10-30 seconds for 1 minute audio
- **GPU-accelerated**: ~2-5 seconds for 1 minute audio
- **Memory usage**: ~2-4GB GPU memory for small model

## Technical Details

### Why This Setup Works

1. **WSL2 GPU Passthrough**: NVIDIA drivers on Windows enable GPU access from WSL2
2. **Library Symlinks**: Versioned symlinks allow PyTorch to find expected cuDNN versions
3. **Environment Variables**: LD_LIBRARY_PATH ensures dynamic linker finds libraries
4. **System Configuration**: /etc/ld.so.conf.d/ makes libraries discoverable system-wide

### File Structure
```
/lib/x86_64-linux-gnu/
├── libcudnn_ops.so.9.10.2 (actual library)
├── libcudnn_ops.so.9.1.0 -> libcudnn_ops.so.9.10.2 (symlink)
├── libcudnn_ops.so.9.1 -> libcudnn_ops.so.9.10.2 (symlink)
└── libcudnn_ops.so.9 -> libcudnn_ops.so.9.10.2 (symlink)
```

## Maintenance

### Updating Libraries
```bash
# Update cuDNN
sudo apt update
sudo apt install --only-upgrade libcudnn9-cuda-12

# Update PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Backup Configuration
```bash
# Backup library configuration
sudo cp /etc/ld.so.conf.d/cudnn.conf ~/cudnn.conf.backup

# Backup bash profile
cp ~/.bashrc ~/.bashrc.backup
```

This setup provides a robust, GPU-accelerated speech-to-text system that will work reliably in WSL2 environments. 