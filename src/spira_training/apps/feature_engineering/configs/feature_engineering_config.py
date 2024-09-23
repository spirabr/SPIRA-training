from pydantic import BaseModel

from src.spira_training.apps.feature_engineering.configs.audio_config import AudioConfig
from src.spira_training.apps.feature_engineering.configs.audio_processor_config import AudioProcessorConfig
from src.spira_training.apps.feature_engineering.configs.audio_feature_transformer_config import AudioFeatureTransformerConfig

class FeatureEngineeringConfig(BaseModel):
    audio: AudioConfig
    audio_processor: AudioProcessorConfig
    audio_feature_transformer: AudioFeatureTransformerConfig