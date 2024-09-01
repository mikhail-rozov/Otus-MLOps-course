import os
import time

from kafka import KafkaConsumer, KafkaProducer
import mlflow
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, hour, udf, floor
from pyspark.sql.types import (StructType, StructField, LongType,
                               IntegerType, FloatType, TimestampType)


TRACKING_SERVER_HOST = "10.0.0.10"
MLFLOW_MODEL_RUN_ID = "1dd03745f01b4d7a81faf9cabe750629" # Run, where the model is
PRODUCER_METRIC_RUN_ID = "e28b7e916fbe4cda89fcee42ed2d4dad" # Run, where the producer metric is being logged
MODEL_URI = f"runs:/{MLFLOW_MODEL_RUN_ID}/hometask_7_model"
BOOTSTRAP_SERVERS = ['rc1a-jt2qmjtbqheks18k.mdb.yandexcloud.net:9091']


def time_bin(hour):
    if 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    elif 18 <= hour < 24:
        return 3  # Evening
    else:
        return 4  # Night


def clean_dataset(df):

    # Cleaning duplicates
    df = df.dropDuplicates(["transaction_id"])

    # # Filling NaN values in terminal_id
    # df = df.fillna({"terminal_id": "0"})

    # Correcting negative values in customer_id
    df = df.withColumn("customer_id", when(col("customer_id") < 0, col("customer_id") * -1)
                       .otherwise(col("customer_id")))

    # # Dropping NaNs in tx_datetime
    # df = df.dropna(subset="tx_datetime")
    
    # Fixing seconds and days mismatch
    df = df.withColumn("tx_time_days", when(floor(col("tx_time_seconds") / 86400) != col("tx_time_days"),
                                            floor(col("tx_time_seconds") / 86400)).otherwise(col("tx_time_days")))
    
    return df


def transform_data(df_cleaned: DataFrame, is_train: bool) -> DataFrame:

    if is_train:
        # Downsample the data, because there is a substantial imbalance in target column values
        df_majority = df_cleaned.filter(col("tx_fraud") == 0)
        df_minority = df_cleaned.filter(col("tx_fraud") == 1)

        df_majority_downsampled = df_majority.sample(withReplacement=False,
                                                     fraction=df_minority.count() / df_majority.count())
        df_cleaned = df_majority_downsampled.union(df_minority)

    transformed_df = df_cleaned.withColumn("is_zero_terminal", when(col("terminal_id") == "0", 1).otherwise(0))

    # Using time from timestamp column in bins as a new attribute
    time_bin_udf = udf(time_bin, IntegerType())
    transformed_df = transformed_df.withColumn("time_of_day", time_bin_udf(hour(col("tx_datetime"))))

    return transformed_df


def main():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    spark = (
        SparkSession
        .builder
        .appName("Otus-hometask")
        .getOrCreate()
    )

    consumer = KafkaConsumer(
        'input',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username='mlops2',
        sasl_plain_password='otus-mlops',
        ssl_cafile="CA.pem",
        fetch_min_bytes=100000,
        fetch_max_wait_ms=3000,
        session_timeout_ms=30000
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username='mlops2',
        sasl_plain_password='otus-mlops',
        ssl_cafile="CA.pem")


    type_mapping = {
        "transaction_id": int,
        "tx_datetime": 'datetime64[ns]',
        "customer_id": int,
        "terminal_id": int,
        "tx_amount": float,
        "tx_time_seconds": int,
        "tx_time_days": int,
        "tx_fraud": int,
        "tx_fraud_scenario": int,
    }

    column_names = ["transaction_id", "tx_datetime", "customer_id", "terminal_id",
                    "tx_amount", "tx_time_seconds", "tx_time_days", "tx_fraud", "tx_fraud_scenario"]

    schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("tx_datetime", TimestampType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("terminal_id", LongType(), True),
        StructField("tx_amount", FloatType(), True),
        StructField("tx_time_seconds", IntegerType(), True),
        StructField("tx_time_days", IntegerType(), True),
        StructField("tx_fraud", IntegerType(), True),
        StructField("tx_fraud_scenario", IntegerType(), True),
    ])

    model = mlflow.spark.load_model(MODEL_URI)

    while True:
        msg_pack = consumer.poll(1000, max_records=30000)
        if len(msg_pack) == 0:
            continue
        
        restart = False

        start_time = time.time()
        data = []
        
        for tp, lines in msg_pack.items():
            for line in lines:
                data.append(line.value.decode("utf-8").split(","))
        
        pandas_df = pd.DataFrame(data, columns=column_names)
        
        # Part of data cleaning process moved here because data types change below causes errors with NaN values 
        pandas_df["terminal_id"] = pandas_df["terminal_id"].fillna("0")
        pandas_df.dropna(subset="tx_datetime", inplace=True)
            
        for column, dtype in type_mapping.items():
            try:
                pandas_df[column] = pandas_df[column].astype(dtype)
            except ValueError:
                pandas_df[column] = pandas_df[column].str.replace("", "0")
                
                # If we encounter unexpected problem with the data, let's just skip this batch for the experiment
                try:
                    pandas_df[column] = pandas_df[column].astype(dtype)
                except ValueError:
                    restart = True
                    break
        if restart:
            continue    

        spark_df = spark.createDataFrame(pandas_df, schema=schema)
        spark_df = clean_dataset(spark_df)
        spark_df = transform_data(spark_df, is_train=False)
        predictions = model.transform(spark_df)
        
        predictions_list = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
        predictions_list = list(map(int, predictions_list))
        predictions_str = "\n".join(map(str, predictions_list))
        
        producer.send('predictions', predictions_str.encode("utf-8"))
        producer.flush()
        
        end_time = time.time()
        transactions_per_second = len(data) / (end_time - start_time)
        mlflow.log_metric("transactions_per_second_processed_and_returned", transactions_per_second, run_id=PRODUCER_METRIC_RUN_ID)
    

if __name__ == "__main__":
    main()
