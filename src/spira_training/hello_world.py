import sys

class HelloWorld:
    def __init__(self):
        self.message = "Hello world (from SPIRA-training)!!"
        self.version = sys.version

    def print_message_and_version(self):
        print(self.message)
        print(self.version)