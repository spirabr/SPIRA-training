from src.spira_training.apps.feature_engineering.configs.audio_processor_config import AudioProcessorType, \
    AudioProcessorConfig
from src.spira_training.shared.adapters.melspectrogram_audio_processor import MelspectrogramAudioProcessor
from src.spira_training.shared.adapters.mfcc_audio_processor import MFCCAudioProcessor
from src.spira_training.shared.adapters.spectrogram_audio_processor import SpectrogramAudioProcessor
from src.spira_training.shared.ports.audio_processor import AudioProcessor


def create_audio_processor(config: AudioProcessorConfig) -> AudioProcessor:

    match config.feature_type:
        case AudioProcessorType.MFCC:
            return MFCCAudioProcessor(
                config.mfcc,
                config.hop_length,
            )
        case AudioProcessorType.SPECTROGRAM:
                return SpectrogramAudioProcessor(
                config.spectrogram,
                config.hop_length,
            )
        case AudioProcessorType.MELSPECTROGRAM:
            return MelspectrogramAudioProcessor(
                config.melspectrogram,
                config.hop_length,
            )
        case _:
            raise ValueError(f"Unknown audio processor type: {config.feature_type}")
