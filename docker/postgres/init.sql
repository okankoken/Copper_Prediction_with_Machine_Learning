CREATE DATABASE airflow_db OWNER copper_user;
CREATE DATABASE mlflow_db OWNER copper_user;

\connect copper_ml

CREATE EXTENSION IF NOT EXISTS vector;