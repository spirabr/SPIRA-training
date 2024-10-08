# test_hello_world.py
import sys
from io import StringIO

from src.spira_training.hello_world import HelloWorld


def test_hello_world():
    mockOutput = StringIO()  # substituir a saída padrão por uma string

    hello = HelloWorld(output=mockOutput)

    hello.print_message_and_version()

    output = mockOutput.getvalue().lower()

    assert "hello world" in output
    assert f"{sys.version}".lower() in output
