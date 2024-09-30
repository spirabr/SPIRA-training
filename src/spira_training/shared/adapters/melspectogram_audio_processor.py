import torchaudio.transforms as transforms

from src.spira_training.shared.ports.audio_processor import AudioProcessor


class MelspectogramAudioProcessor(AudioProcessor):

    def __init__(self, config: 'MelspectrogramAudioProcessorConfig', hop_length: int):
        self.hop_length = hop_length
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=hop_length,
            n_mels=config.num_mels,
        )

    def get_transformer(self):
        return self.melspectrogram
