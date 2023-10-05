import librosa
from IPython.display import Audio, display
from pyannote.audio import Audio
from pyannote.core import Segment
import pyannote.audio
import torch

audio = Audio()

def segment_embedding(segment, audio_path, duration):
    # Get the start and end timestamps for the segment
    start = segment["start"]

    # Pretrained Speaker Embedding Model
    embedding_model = pyannote.audio.pipelines.speaker_verification.PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))  # Use CPU instead of CUDA

    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_path, clip)
    return embedding_model(waveform[None])