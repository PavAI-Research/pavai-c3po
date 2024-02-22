import os

import pika

QUEUES = ["start_fetch", "data_processing"]

username = os.environ.get("RABBITMQ_USER", "DEV_USER")
password = os.environ.get("RABBITMQ_PASSWORD", "CHANGE_ME")
host = os.environ.get("RABBITMQ_HOST", "localhost")


def open_channel():
    """
    Opens a connection, a channel, creates queues and then returns this to the caller.
    """

    # Create the connection
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, credentials=credentials)
    )
    channel = connection.channel()

    # Create all the queues
    for queue in QUEUES:
        channel.queue_declare(queue=queue, durable=True)

    return channel

