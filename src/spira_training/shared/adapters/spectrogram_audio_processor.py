import torchaudio.transforms as transforms

from src.spira_training.apps.feature_engineering.configs.audio_processor_config import SpectrogramAudioProcessorConfig
from src.spira_training.shared.ports.audio_processor import AudioProcessor


class SpectrogramAudioProcessor(AudioProcessor):

    def __init__(self, config: 'SpectrogramAudioProcessorConfig', hop_length: int):
        self.hop_length = hop_length
        self.spectrogram = transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=hop_length,
            power=config.power,
        )

    def get_transformer(self):
        return self.spectrogram
