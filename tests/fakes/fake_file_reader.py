from src.spira_training.shared.ports.file_reader import FileReader

class FakeFileReader(FileReader):
    def read(self, path: str) -> list[str]:
        return ["path/to/audio1.wav", "path/to/audio2.wav"]