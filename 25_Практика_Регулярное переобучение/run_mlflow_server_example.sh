# pip install boto3
# pip install psycopg2-binary

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=example
export AWS_SECRET_ACCESS_KEY=example

mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri postgresql://otus_mlflow_database_user:otus_mlflow_database_password@10.0.0.11:5432/mlflow_database \
    --registry-store-uri postgresql://otus_mlflow_database_user:otus_mlflow_database_password@10.0.0.11:5432/mlflow_database \
    --default-artifact-root s3://mrozov-mlops/mlflow-artifacts \
    --no-serve-artifacts
