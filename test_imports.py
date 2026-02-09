import torch
import torchaudio
from diffusers import DiffusionPipeline
import soundfile as sf
import librosa
import numpy as np


print("All packages imported successfully!")
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)

import tempfile
import io
print("Audio packages ready")