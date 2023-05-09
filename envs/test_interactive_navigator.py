import random

from dep_cannoli_streaming_client import CannoliStreamingClient
import json
import pprint


def main():
    c = CannoliStreamingClient('spray_and_pray.c')

    for i in range(10):
        _step(c)

    c.reset()


def _step(client):
    random_action = random.choice(['1', '2', '3', '4', '5', '6'])
    client.write_to_cannoli(random_action)

    data = client.read_from_cannoli()
    pprint.pprint(json.loads(data))
