import os

import pymongo

# Credentials for Mongo
username = os.environ.get("MONGO_USER", "DEV_USER")
password = os.environ.get("MONGO_PASSWORD", "CHANGE_ME")
host = os.environ.get("MONGO_HOST", "localhost")
port = os.environ.get("MONGO_PORT", 27017)


def open_db():
    """
    A function to get a connection to the mongo database.
    """

    connection = pymongo.MongoClient(
        host=host,
        port=port,
        username=username,
        password=password,
    )

    db = connection["pavai"]

    return db