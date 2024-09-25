from typing import ClassVar

from pydantic import BaseModel, ConfigDict

class AudioProcessorType(BaseModel):
    MFCC: ClassVar[str] = "mfcc"
    SPECTROGRAM: ClassVar[str] = "spectrogram"
    MELSPECTROGRAM: ClassVar[str] = "melspectrogram"

class MFCCAudioProcessorConfig(BaseModel):
    sample_rate: int
    num_mels: int
    num_mfcc: int
    log_mels: bool
    n_fft: int
    win_length: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

class AudioProcessorConfig(BaseModel):
    feature_type: AudioProcessorType
    mfcc: MFCCAudioProcessorConfig
    spectrogram: SpectrogramAudioProcessorConfig
    melspectrogram: MelspectrogramAudioProcessorConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)