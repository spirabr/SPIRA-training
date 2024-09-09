from pydantic import BaseModel

from audio_config import AudioConfig
from audio_processor_config import AudioProcessorConfig
from audio_feature_transformer_config import AudioFeatureTransformerConfig

class FeatureEngineeringConfig(BaseModel):
    audio: AudioConfig
    audio_processor: AudioProcessorConfig
    audio_feature_transformer: AudioFeatureTransformerConfig