import fastapi

from pavai.finetune.db_storage import open_db
##from pavai.finetune.worker import start_fetch_job
from pavai.finetune.worker import fetch_worker_runner

app = fastapi.FastAPI()

@app.get("/finetune")
def get_latest_data_record() -> dict:
    """Gets the latest processed job record."""

    db = open_db()

    record = db.results.find_one(sort=[("_id", -1)])

    if record:
        # Drop the Mongo ID field
        if "_id" in record:
            del record["_id"]
        return record

    else:
        # Empty database
        return {}


@app.post("/refresh")
def refresh_data() -> dict:
    """Starts a background data refresh job."""
    ##start_fetch_job()
    fetch_worker_runner.start_fetch_job()

    return {"status": "ok"}
