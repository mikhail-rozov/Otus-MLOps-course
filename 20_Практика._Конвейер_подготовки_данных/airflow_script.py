import uuid
from datetime import datetime, timedelta
from airflow import DAG, settings
from airflow.models import Connection, Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.yandex.operators.yandexcloud_dataproc import (
    DataprocCreateClusterOperator,
    DataprocCreatePysparkJobOperator,
    DataprocDeleteClusterOperator,
)

# Common settings for environment
YC_DP_FOLDER_ID = Variable.get("FOLDER_ID")
YC_DP_SUBNET_ID = Variable.get("SUBNET_ID")
YC_DP_SA_ID = Variable.get("SA_ID")
YC_DP_AZ = 'ru-central1-a'
YC_DP_SSH_PUBLIC_KEY = Variable.get("SSH_PUBLIC")
YC_DP_GROUP_ID = Variable.get("SERVICE_GROUP_ID")
AWS_ACCESS_KEY = Variable.get("S3_KEY_ID")
AWS_SECRET_KEY = Variable.get("S3_SECRET_KEY")

# Settings for S3 buckets
YC_SOURCE_BUCKET = 'mrozov-mlops'                           # YC S3 bucket for pyspark source files
YC_DP_LOGS_BUCKET = YC_SOURCE_BUCKET + '/airflow_logs/'     # YC S3 bucket for Data Proc cluster logs

# Create Object Storage connection
session = settings.Session()
ycS3_connection = Connection(
    conn_id='yc-s3',
    conn_type='s3',
    host='https://storage.yandexcloud.net/',
    extra={
        "aws_access_key_id": AWS_ACCESS_KEY,
        "aws_secret_access_key": AWS_SECRET_KEY,
        "host": "https://storage.yandexcloud.net/"
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycS3_connection.conn_id).first():
    session.add(ycS3_connection)
    session.commit()

# Create service account connection
ycSA_connection = Connection(
    conn_id='yc-SA',
    conn_type='yandexcloud',
    extra={
        "extra__yandexcloud__public_ssh_key": Variable.get("DP_PUBLIC_SSH_KEY"),
        "extra__yandexcloud__service_account_json_path": Variable.get("DP_SA_PATH")
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycSA_connection.conn_id).first():
    session.add(ycSA_connection)
    session.commit()

# DAG settings
with DAG(
        dag_id='DATA_PREPROCESSING',
        start_date=datetime(year=2024, month=6, day=16),
        schedule_interval=timedelta(minutes=15),  # '@daily' for everyday start, 15 minutes for faster necessary 3 runs
        catchup=False,
        max_active_runs=1
) as ingest_dag:

    # Create cluster
    create_spark_cluster = DataprocCreateClusterOperator(
        task_id='dp-cluster-create-task',
        folder_id=YC_DP_FOLDER_ID,
        cluster_name=f'tmp-dp-{uuid.uuid4()}',
        cluster_description='Temporary cluster for Spark processing under Airflow orchestration',
        subnet_id=YC_DP_SUBNET_ID,
        s3_bucket=YC_DP_LOGS_BUCKET,
        service_account_id=YC_DP_SA_ID,
        ssh_public_keys=YC_DP_SSH_PUBLIC_KEY,
        zone=YC_DP_AZ,
        cluster_image_version='2.0',
        masternode_resource_preset='s3-c2-m8',
        masternode_disk_type='network-ssd',
        masternode_disk_size=40,
        datanode_resource_preset='s3-c4-m16',
        datanode_disk_type='network-ssd',
        datanode_disk_size=128,
        datanode_count=3,
        services=['YARN', 'SPARK', 'HDFS', 'MAPREDUCE'],  
        computenode_count=0,           
        connection_id=ycSA_connection.conn_id,
        dag=ingest_dag
    )

    # Run PySpark job
    poke_spark_processing = DataprocCreatePysparkJobOperator(
        task_id='dp-cluster-pyspark-task',
        main_python_file_uri=f's3a://{YC_SOURCE_BUCKET}/scripts/data_cleaning.py',
        connection_id=ycSA_connection.conn_id,
        dag=ingest_dag,
        properties={
            "spark.yarn.appMasterEnv.AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY,
            "spark.yarn.appMasterEnv.AWS_SECRET_ACCESS_KEY": AWS_SECRET_KEY
        }
    )

    # Delete cluster
    delete_spark_cluster = DataprocDeleteClusterOperator(
        task_id='dp-cluster-delete-task',
        trigger_rule=TriggerRule.ALL_DONE,
        dag=ingest_dag
    )

    # DAG direction
    create_spark_cluster >> poke_spark_processing >> delete_spark_cluster
