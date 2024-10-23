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

def concatenate_audios(audios: AudioCollection | GeneratedAudioCollection) -> Audio:
    if len(audios) == 0:
        return Audio(wav=create_empty_wav(), sample_rate=0)

    if _check_audios_have_different_sample_rate(audios):
        raise ValueError("Sample rates are not equal")

    wav_list = [audio.wav for audio in audios]
    concatenated_wav = concatenate_wavs(wav_list)

    return Audio(concatenated_wav, sample_rate=audios[0].sample_rate)

def _check_audios_have_different_sample_rate(audios: GeneratedAudioCollection) -> bool:
    sample_rates = {audio.sample_rate for audio in audios}
    return len(sample_rates) > 1


def add_padding_to_audio_collection(audios: AudioCollection) -> AudioCollection:
    max_audio_length = audios.get_max_audio_length()

    return AudioCollection(
        [audio.add_padding(max_audio_length) for audio in audios],
        audios.hop_length
    )
