import torchaudio.transforms as transforms

from src.spira_training.apps.feature_engineering.configs.audio_processor_config import MFCCAudioProcessorConfig
from src.spira_training.shared.ports.audio_processor import AudioProcessor


class MFCCAudioProcessor(AudioProcessor):
    def __init__(self, config: 'MFCCAudioProcessorConfig', hop_length: int):
        self.hop_length = hop_length
        self.mfcc = transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.num_mfcc,
            log_mels=config.log_mels,
            melkwargs={
                "n_fft": config.n_fft,
                "win_length": config.win_length,
                "hop_length": hop_length,
                "n_mels": config.num_mels,
            },
        )

    def get_transformer(self):
        return self.mfcc
