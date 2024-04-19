import sys

class HelloWorld:
    def __init__(self, output):
        self.message = "Hello world (from SPIRA-training)!!"
        self.version = sys.version
        self.output = output

    def print_message_and_version(self):
        print(self.message, file=self.output)
        print(self.version, file=self.output)