from src.spira_training.apps.feature_engineering.configs.audio_config import AudioConfig, DatasetPaths
from src.spira_training.apps.feature_engineering.configs.audio_feature_transformer_config import AudioFeatureTransformerConfig, AudioFeatureTransformerOptions, AudioFeatureTransformersCollection, NoisyAudioFeatureTransformerConfig, OverlappedAudioFeatureTransformerConfig, MixedAudioFeatureTransformerConfig
from src.spira_training.apps.feature_engineering.configs.audio_processor_config import AudioProcessorConfig, AudioProcessorType, MFCCAudioProcessorConfig, SpectrogramAudioProcessorConfig, MelspectrogramAudioProcessorConfig
from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import FeatureEngineeringConfig

def make_feature_engineering_config():
    return FeatureEngineeringConfig(
        audio=AudioConfig(
            dataset_paths=DatasetPaths(
                patients_csv="patients.csv",
                controls_csv="controls.csv",
                noises_csv="noises.csv"
            ),
            hop_length=512,
            normalize=True
        ),
        audio_processor=AudioProcessorConfig(
            feature_type=AudioProcessorType.MFCC,
            mfcc=MFCCAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                win_length=400
            ),
            spectrogram=SpectrogramAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                mel_fmin=0.0,
                mel_fmax=8000.0,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                num_freq=257,
                win_length=400
            ),
            melspectrogram=MelspectrogramAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                mel_fmin=0.0,
                mel_fmax=8000.0,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                num_freq=257,
                win_length=400
            )
        ),
        audio_feature_transformer=AudioFeatureTransformerConfig(
            options=AudioFeatureTransformerOptions(
                use_noise=True,
                use_overlapping=True,
                use_padding=True,
                use_mixture=True
            ),
            audio_feature_transformers=AudioFeatureTransformersCollection(
                noisy_audio=NoisyAudioFeatureTransformerConfig(
                    num_noise_control=5,
                    num_noise_patient=5,
                    noise_max_amp=0.5,
                    noise_min_amp=0.1
                ),
                overlapped_audio=OverlappedAudioFeatureTransformerConfig(
                    window_length=400,
                    step_size=160
                ),
                mixed_audio=MixedAudioFeatureTransformerConfig(
                    alpha=0.5,
                    beta=0.5
                )
            )
        )
    )