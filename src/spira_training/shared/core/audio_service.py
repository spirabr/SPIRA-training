from typing import Iterable

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_collection import AudioCollection
from src.spira_training.shared.core.models.generated_audio_collection import GeneratedAudioCollection
from src.spira_training.shared.core.models.wav import create_empty_wav, concatenate_wavs

def create_slices_from_audio(audio: Audio, window_length: int, step_size: int) -> GeneratedAudioCollection:
    slices = []
    slice_index = 0

    while slice_index < len(audio):
        start = slice_index
        end = slice_index + window_length

        if end > len(audio):
            end = len(audio)

        slices.append(_create_slice(start, end))
        slice_index += step_size

    return GeneratedAudioCollection(generated_audios=slices)

def _create_slice(self, start_index: int, end_index: int) -> 'Audio':
    if start_index < 0 or end_index < 0 or start_index >= end_index:
        raise ValueError(f"Invalid range [{start_index}:{end_index}]")

    return Audio(
        wav=self.wav.slice(
            # Audios are indexed in sample_rate chunks
            start_index=start_index * self.sample_rate,
            end_index=end_index * self.sample_rate,
        )
    )

def concatenate_audios(audio_collection: AudioCollection | GeneratedAudioCollection) -> Audio:
    if len(audio_collection) == 0:
        return Audio(wav=create_empty_wav(), sample_rate=0)

    if _check_audios_have_different_sample_rate(audio_collection):
        raise ValueError("Sample rates are not equal")

    wav_list = [audio.wav for audio in audio_collection]
    concatenated_wav = concatenate_wavs(wav_list)

    return Audio(concatenated_wav, sample_rate=audio_collection[0].sample_rate)

def _check_audios_have_different_sample_rate(audio_collection: GeneratedAudioCollection) -> bool:
    sample_rates = {audio.sample_rate for audio in audio_collection}
    return len(sample_rates) > 1


def add_padding_to_audio_collection(audio_collection: AudioCollection) -> AudioCollection:
    max_audio_length = audio_collection.get_max_audio_length()

    return AudioCollection(
        [audio.add_padding(max_audio_length) for audio in audio_collection],
        audio_collection.hop_length
    )

def get_pairs_of_audios(audio_collection: AudioCollection) -> Iterable[tuple[Audio, Audio]]:
    iterator = iter(audio_collection.audio_list)
    while True:
        try:
            yield next(iterator), next(iterator)
        except StopIteration:
            break

def mix_audios(self, first: Audio, second: Audio) -> tuple[Audio, Audio]:
    probability = self.randomizer.get_probability(self.alpha, self.beta)
    new_first = first * probability + second * (1 - probability)
    new_second = first * (1 - probability) + second * probability
    return new_first, new_second