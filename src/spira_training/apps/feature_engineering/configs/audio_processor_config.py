from pydantic import BaseModel


class AudioProcessorType(BaseModel):
    MFCC = "mfcc"
    SPECTROGRAM = "spectrogram"
    MELSPECTROGRAM = "melspectrogram"

class MFCCAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    num_mfcc: int
    log_mels: bool
    n_fft: int
    win_length: int

class SpectrogramAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: None
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    win_length: int

class MelspectrogramAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: None
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