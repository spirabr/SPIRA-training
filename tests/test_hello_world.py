# test_hello_world.py
import sys
from io import StringIO
import pytest
from src.spira_training.hello_world import HelloWorld

def test_hello_world():
    # guardar a saída padrão original
    stdout_original = sys.stdout
    sys.stdout = StringIO()  # substituir a saída padrão por uma string

    hello = HelloWorld()  
    hello.print_message_and_version()  

    output = sys.stdout.getvalue()  

    sys.stdout = stdout_original

    assert output == "Hello world (from SPIRA-training)!!\n" + sys.version + "\n"