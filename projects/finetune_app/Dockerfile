FROM python:3.11-slim

ARG wheel=finetune_app-0.1.0-py3-none-any.whl

WORKDIR /code
COPY ./dist/$wheel /code/$wheel

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir /code/$wheel

CMD ["uvicorn", "pavai.finetune_api:app", "--host", "0.0.0.0", "--port", "8008"]
