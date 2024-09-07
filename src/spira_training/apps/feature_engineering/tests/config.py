class TestConfig:
    class Paths:
        dataset = "test_dataset_path"

    class Parameters:
        class Dataset:
            patients_csv = "test_patients_csv"
            controls_csv = "test_controls_csv"
            noises_csv = "test_noises_csv"
            normalize = True

        class Audio:
            hop_length = 512

        class FeatureEngineering:
            class NoisyAudio:
                num_noise_control = 5

    paths = Paths()
    parameters = Parameters()

class TestRandomizer:
    pass

class TestDatasetRepository:
    def __init__(self):
        self.saved_datasets = []

    def save_dataset(self, dataset, path):
        self.saved_datasets.append((dataset, path))