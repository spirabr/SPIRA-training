import pytest

from src.spira_training.shared.adapters.audio_processor.mfcc_audio_processor import MFCCAudioProcessor
from src.spira_training.shared.core.audio_processor_factory import create_audio_processor
from tests.fakes.fake_feature_engineering_config import make_feature_engineering_config


@pytest.mark.asyncio
async def test_create_audio_processor_mfcc():
    # Arrange
    config = make_feature_engineering_config()

    # Act
    audio_processor = create_audio_processor(config.audio_processor)

    # Assert
    assert isinstance(audio_processor, MFCCAudioProcessor)