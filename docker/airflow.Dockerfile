ARG AIRFLOW_VERSION

FROM apache/airflow:${AIRFLOW_VERSION}

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    -r /requirements.txt
