import pytest

from src.spira_training.apps.feature_engineering.configs.audio_processor_config import (
    AudioProcessorType,
)
from src.spira_training.shared.adapters.feature_transformer.mfcc_feature_transformer import (
    MFCCFeatureTransformer,
)
from src.spira_training.shared.core.audio_processor_factory import (
    create_audio_processor,
)
from tests.fakes.fake_feature_engineering_config import make_audio_processor_config
from tests.fakes.fake_pytorch_audio_factory import FakePytorchTensorFactory


@pytest.mark.asyncio
async def test_create_audio_processor_mfcc():
    # Arrange
    audio_processor_config = make_audio_processor_config(AudioProcessorType.MFCC)

    # Act
    audio_processor = create_audio_processor(
        audio_processor_config, FakePytorchTensorFactory()
    )

    # Assert
    assert isinstance(audio_processor.feature_transformer, MFCCFeatureTransformer)
