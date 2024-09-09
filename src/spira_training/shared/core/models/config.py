from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import FeatureEngineeringConfig
from src.spira_training.apps.model_evaluation.model_evaluation_config import ModelEvaluationConfig
from src.spira_training.apps.model_publish.model_publish_config import ModelPublishConfig


class Config:
    feature_engineering_config: FeatureEngineeringConfig
    model_trainer_config: ModelTrainerConfig
    model_evaluation_config: ModelEvaluationConfig
    model_publish_config: ModelPublishConfig