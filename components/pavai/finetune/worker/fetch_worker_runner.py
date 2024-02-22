import json

from pavai.finetune.job_data import fetch_data
from pavai.finetune.job_queue import open_channel

LISTEN_QUEUE = "start_fetch"
PUSH_QUEUE = "data_processing"


def callback(ch, method, properties, body):
    """
    Fetches data and hands it off to the next queue.
    """

    print("Fetch data job started...")

    # Fall the fetch data component to create and fetch data
    data = fetch_data()

    # Send data to the next queue
    ch.basic_publish(exchange="", routing_key=PUSH_QUEUE, body=json.dumps(data))

    # Acknowledge the incoming message to remove it from the queue
    ch.basic_ack(delivery_tag=method.delivery_tag)

    print("Done. Data sent to to queue for data processing.")


def run_fetch_worker():
    """
    Sets up and runs the fetch worker 1.
    Creates a blocking connection so remains running.
    """
    print(f"run_fetch_worker is listening on queue [{LISTEN_QUEUE}]")
    
    # Get a new channel from the base
    channel = open_channel()

    # Register the callback
    channel.basic_consume(queue=LISTEN_QUEUE, on_message_callback=callback)

    # This is a blocking connection
    channel.start_consuming()
    

def start_fetch_job():
    """Starts the first worker."""
    
    # Get a new channel from the base
    channel = open_channel()

    # Send an empty message to the queue
    channel.basic_publish(exchange="", routing_key=LISTEN_QUEUE, body="")

    print("New fetch job requested")

    # Close cleanly
    channel.close()