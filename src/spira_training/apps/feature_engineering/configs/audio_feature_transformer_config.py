from pydantic import BaseModel


class AudioFeatureTransformerOptions(BaseModel):
    use_noise: bool
    use_overlapping: bool
    use_padding: bool
    use_mixture: bool

class NoisyAudioFeatureTransformerConfig(BaseModel):
    num_noise_control: int
    num_noise_patient: int
    noise_max_amp: float
    noise_min_amp: float

class OverlappedAudioFeatureTransformerConfig(BaseModel):
    window_length: int
    step_size: int

class MixedAudioFeatureTransformerConfig(BaseModel):
    alpha: float
    beta: float

class AudioFeatureTransformersCollection(BaseModel):
    noisy_audio: NoisyAudioFeatureTransformerConfig
    overlapped_audio: OverlappedAudioFeatureTransformerConfig
    mixed_audio: MixedAudioFeatureTransformerConfig

class AudioFeatureTransformerConfig(BaseModel):
    options: AudioFeatureTransformerOptions
    audio_feature_transformers: AudioFeatureTransformersCollection