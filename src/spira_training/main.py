import sys
from spira_training.hello_world import HelloWorld

greetter = HelloWorld(
    output=sys.stdout
)

greetter.print_message_and_version()