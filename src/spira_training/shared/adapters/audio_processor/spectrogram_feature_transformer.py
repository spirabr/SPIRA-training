import torchaudio.transforms as transforms

from src.spira_training.apps.feature_engineering.configs.audio_processor_config import SpectrogramAudioProcessorConfig
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.feature_transformer import FeatureTransformer


class SpectrogramFeatureTransformer(FeatureTransformer):
    def __init__(self, config: 'SpectrogramAudioProcessorConfig', hop_length: int):
        self.spectrogram = transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=hop_length,
            power=config.power,
        )

    def transform(self, wav: Wav) -> Wav:
        return self.spectrogram(wav)
