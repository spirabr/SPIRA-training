from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

class AudioProcessorType(str, Enum):
    MFCC = "mfcc"
    SPECTROGRAM = "spectrogram"
    MELSPECTROGRAM = "melspectrogram"

class MFCCAudioProcessorConfig(BaseModel):
    sample_rate: int
    num_mels: int
    num_mfcc: int
    log_mels: bool
    n_fft: int
    win_length: int

class SpectrogramAudioProcessorConfig(BaseModel):
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: float
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    win_length: int

class MelspectrogramAudioProcessorConfig(BaseModel):
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: float
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    win_length: int

class AudioProcessorConfig(BaseModel):
    feature_type: AudioProcessorType
    mfcc: MFCCAudioProcessorConfig
    spectrogram: SpectrogramAudioProcessorConfig
    melspectrogram: MelspectrogramAudioProcessorConfig
