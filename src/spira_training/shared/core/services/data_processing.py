from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.ports.audios_repository import AudiosRepository


def load_data(config):

    #audios = AudiosRepository.get_audios(config.paths)

    # patients_paths = read_valid_paths_from_csv(config.parameters.dataset.patients_csv)
    # controls_paths = read_valid_paths_from_csv(config.parameters.dataset.controls_csv)
    # noises_paths = read_valid_paths_from_csv(config.parameters.dataset.noises_csv)
    #
    # patients_inputs = Audios.load(
    #     patients_paths,
    #     config.parameters.audio.hop_length,
    #     config.parameters.dataset.normalize,
    # )
    # controls_inputs = Audios.load(
    #     controls_paths,
    #     config.parameters.audio.hop_length,
    #     config.parameters.dataset.normalize,
    # )
    # noises = Audios.load(
    #     noises_paths,
    #     config.parameters.audio.hop_length,
    #     config.parameters.dataset.normalize,
    # )
    #
    # return patients_inputs, controls_inputs, noises
    pass


def generate_dataset(config, randomizer, patients_inputs, controls_inputs, noises):
    # audio_processor = create_audio_processor(config.parameters.audio)
    #
    # patient_feature_transformer = create_audio_feature_transformer(
    #     randomizer,
    #     audio_processor,
    #     config.options.feature_engineering,
    #     config.parameters.feature_engineering,
    #     config.parameters.feature_engineering.noisy_audio.num_noise_control,
    #     noises,
    # )
    #
    # control_feature_transformer = create_audio_feature_transformer(
    #     randomizer,
    #     audio_processor,
    #     config.options.feature_engineering,
    #     config.parameters.feature_engineering,
    #     config.parameters.feature_engineering.noisy_audio.num_noise_control,
    #     noises,
    # )

    # patients_features = patient_feature_transformer.transform_into_features(patients_inputs)
    # controls_features = control_feature_transformer.transform_into_features(controls_inputs)
    #
    # patients_label = [Label.POSITIVE for _ in range(len(patients_features))]
    # controls_label = [Label.NEGATIVE for _ in range(len(controls_features))]
    #
    # features = patients_features + controls_features
    # labels = patients_label + controls_label
    #
    # dataset = Dataset(features=features, labels=labels)

    # return dataset
    pass
