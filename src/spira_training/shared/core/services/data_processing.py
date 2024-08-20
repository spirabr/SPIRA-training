from src.spira_training.shared.adapters.valid_path import read_valid_paths_from_csv
from src.spira_training.shared.core.domain.audio import Audios
from src.spira_training.shared.core.services.audio_feature_transformer import create_audio_feature_transformer
from src.spira_training.shared.core.services.audio_processor import create_audio_processor

from src.spira_training.shared.core.models.transformed_data import TransformedData


def load_and_transform_data(config, randomizer):
    patients_inputs, controls_inputs, noises = load_data(config)

    return transform_data(config, randomizer, patients_inputs, controls_inputs, noises)


def load_data(config):
    patients_paths = read_valid_paths_from_csv(config.parameters.dataset.patients_csv)
    controls_paths = read_valid_paths_from_csv(config.parameters.dataset.controls_csv)
    noises_paths = read_valid_paths_from_csv(config.parameters.dataset.noises_csv)

    patients_inputs = Audios.load(
        patients_paths,
        config.parameters.audio.hop_length,
        config.parameters.dataset.normalize,
    )
    controls_inputs = Audios.load(
        controls_paths,
        config.parameters.audio.hop_length,
        config.parameters.dataset.normalize,
    )
    noises = Audios.load(
        noises_paths,
        config.parameters.audio.hop_length,
        config.parameters.dataset.normalize,
    )

    return patients_inputs, controls_inputs, noises


def transform_data(config, randomizer, patients_inputs, controls_inputs, noises):
    audio_processor = create_audio_processor(config.parameters.audio)

    patient_feature_transformer = create_audio_feature_transformer(
        randomizer,
        audio_processor,
        config.options.feature_engineering,
        config.parameters.feature_engineering,
        config.parameters.feature_engineering.noisy_audio.num_noise_control,
        noises,
    )

    control_feature_transformer = create_audio_feature_transformer(
        randomizer,
        audio_processor,
        config.options.feature_engineering,
        config.parameters.feature_engineering,
        config.parameters.feature_engineering.noisy_audio.num_noise_control,
        noises,
    )

    patients_features = patient_feature_transformer.transform_into_features(patients_inputs)
    controls_features = control_feature_transformer.transform_into_features(controls_inputs)

    patients_label = [1 for _ in range(len(patients_features))]
    controls_label = [0 for _ in range(len(controls_features))]

    features = patients_features + controls_features
    labels = patients_label + controls_label

    transformed_data = TransformedData(features=features, labels=labels)

    return transformed_data
