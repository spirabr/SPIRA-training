import torchaudio.transforms as transforms

from src.spira_training.apps.feature_engineering.configs.audio_processor_config import MFCCAudioProcessorConfig
from src.spira_training.shared.adapters.pytorch_wav import PytorchWav
from src.spira_training.shared.ports.feature_transformer import FeatureTransformer


class MFCCFeatureTransformer(FeatureTransformer):
    def __init__(self, config: 'MFCCAudioProcessorConfig', hop_length: int):
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

    def transform(self, wav: PytorchWav) -> PytorchWav:
        return self.mfcc(wav)
