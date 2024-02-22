FROM python:3.11-slim

ARG wheel=finetune_worker1-0.1.0-py3-none-any.whl

WORKDIR /code
COPY ./dist/$wheel /code/$wheel

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir /code/$wheel

CMD ["python", "-m", "pavai.finetune.worker.fetch_worker_runner_start"]
