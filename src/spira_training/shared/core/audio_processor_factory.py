from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchTensorFactory,
)
from src.spira_training.apps.feature_engineering.configs.audio_processor_config import (
    AudioProcessorType,
    AudioProcessorConfig,
)
from src.spira_training.shared.adapters.feature_transformer.melspectrogram_feature_transformer import (
    MelspectrogramFeatureTransformer,
)
from src.spira_training.shared.adapters.feature_transformer.mfcc_feature_transformer import (
    MFCCFeatureTransformer,
)
from src.spira_training.shared.adapters.feature_transformer.spectrogram_feature_transformer import (
    SpectrogramFeatureTransformer,
)
from src.spira_training.shared.core.audio_processor import AudioProcessor
from src.spira_training.shared.ports.feature_transformer import FeatureTransformer


def create_audio_processor(
    config: AudioProcessorConfig, pytorch_tensor_factory: PytorchTensorFactory
) -> AudioProcessor:
    feature_transformer = create_feature_transformer(config)

    return AudioProcessor(
        feature_transformer=feature_transformer,
        pytorch_tensor_factory=pytorch_tensor_factory,
    )


def create_feature_transformer(config: AudioProcessorConfig) -> FeatureTransformer:
    match config.feature_type:
        case AudioProcessorType.MFCC:
            return MFCCFeatureTransformer(
                config.mfcc,
                config.hop_length,
            )
        case AudioProcessorType.SPECTROGRAM:
            return SpectrogramFeatureTransformer(config.spectrogram, config.hop_length)
        case AudioProcessorType.MELSPECTROGRAM:
            return MelspectrogramFeatureTransformer(
                config.melspectrogram,
                config.hop_length,
            )
        case _:
            raise ValueError(f"Unknown audio processor type: {config.feature_type}")
