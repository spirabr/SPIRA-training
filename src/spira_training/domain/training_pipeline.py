from ports.training_data_loader import TrainingDataLoader

class TrainingPipeline:
    def __init__(
        self,
        training_data_loader : TrainingDataLoader,
        feature_engineering,
        dataset_generator,
        model_trainer,
        model_validator,
        result_publisher
    ):
        self.training_data_loader = training_data_loader
        self.feature_engineering = feature_engineering
        self.dataset_generator = dataset_generator
        self.model_trainer = model_trainer
        self.model_validator = model_validator
        self.result_publisher = result_publisher

    def execute(self):
        training_data = self.training_data_loader.load_data()
        features = self.feature_engineering.extract_features(training_data)
        dataset = self.dataset_generator.generate(features)
        model = self.model_trainer.train(dataset)
        validation_result = self.model_validator.validate(model)
        self.result_publisher.publish(validation_result)
