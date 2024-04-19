# test_hello_world.py
import sys
from io import StringIO
import pytest
from spira_training.hello_world import HelloWorld

def test_hello_world():
    mockOutput = StringIO()  # substituir a saída padrão por uma string

    hello = HelloWorld(
        output=mockOutput
    )  

    hello.print_message_and_version()  

    output = mockOutput.getvalue()  

    assert output.find("Hello world from SPIRA-training)!!")
    assert output.find(f"{sys.version}")