import os
import pickle

import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, floor
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, TimestampType

BUCKET_NAME = "mrozov-mlops"
_BUCKET_PATH = "s3a://" + BUCKET_NAME + "/"
PROCESSED_TXTS_PICKLE = "processed_txts.pkl"
NEW_DATA_PROCESSED_LOG = "data_cleaning_status.log"
N_PARTITIONS = 12
CLEANED_DATASET_PARQUET = "cleaned_dataset_sample.parquet"
NEW_DATA_PARQUET = "new_data.parquet"
OLD_DATA_PARQUET = "old_data.parquet"


def clean_dataset(df):

    # Cleaning duplicates
    df = df.dropDuplicates(["transaction_id"])

    # Filling NaN values in terminal_id
    df = df.fillna({"terminal_id": "0"})

    # Correcting negative values in customer_id
    df = df.withColumn("customer_id", when(col("customer_id") < 0, col("customer_id") * -1)
                       .otherwise(col("customer_id")))

    # Dropping NaNs in tx_datetime
    df = df.dropna(subset="tx_datetime")
    
    # Fixing seconds and days mismatch
    df = df.withColumn("tx_time_days", when(floor(col("tx_time_seconds") / 86400) != col("tx_time_days"),
                                            floor(col("tx_time_seconds") / 86400)).otherwise(col("tx_time_days")))
    
    return df


def main():
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    boto3_session = boto3.session.Session()
    s3 = boto3_session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="ru-central1"
    )

    # Loading all .txt files from s3-bucket, searching for already processed dataset and processed .txt files
    all_txts = set()
    df_current_exists = False
    processed_txts_exists = False
    for key in s3.list_objects(Bucket=BUCKET_NAME)['Contents']:
        if key['Key'].endswith(".txt") and len(all_txts) < 3:
            all_txts.add(key['Key'])
        elif not df_current_exists and key['Key'].startswith(CLEANED_DATASET_PARQUET):
            df_current_exists = True
        elif not processed_txts_exists and key['Key'] == PROCESSED_TXTS_PICKLE:
            processed_txts_exists = True
    
    # Loading set of already processed .txt files
    if processed_txts_exists:
        s3.download_file(Bucket=BUCKET_NAME, Key=PROCESSED_TXTS_PICKLE, Filename=PROCESSED_TXTS_PICKLE)
        with open(PROCESSED_TXTS_PICKLE, "rb") as f:
            processed_txts = pickle.load(f)
    else:
        processed_txts = set()

    spark = (
        SparkSession
        .builder
        .appName("Otus-hometask")
        .getOrCreate()
    )

    # Processing should be only for new .txt files
    # Doing this AFTER spark initialization for proper PySpark job run when there are no new txts
    new_txts = all_txts - processed_txts
    if len(new_txts) == 0:
        spark.stop()
        
        # Saving and uploading info that no new data exists
        with open(NEW_DATA_PROCESSED_LOG, "w") as f:
            f.write("no new data")
        s3.upload_file(NEW_DATA_PROCESSED_LOG, BUCKET_NAME, NEW_DATA_PROCESSED_LOG)

        return

    schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("tx_datetime", TimestampType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("terminal_id", IntegerType(), True),
        StructField("tx_amount", FloatType(), True),
        StructField("tx_time_seconds", IntegerType(), True),
        StructField("tx_time_days", IntegerType(), True),
        StructField("tx_fraud", IntegerType(), True),
        StructField("tx_fraud_scenario", IntegerType(), True),
    ])

    # Writing current dataset to a new file to be able to overwrite the current cleaned file later
    if df_current_exists:
        df_current = spark.read.parquet(_BUCKET_PATH + CLEANED_DATASET_PARQUET)
        (
            df_current
            .repartition(N_PARTITIONS)
            .write 
            .mode("overwrite")
            .parquet(_BUCKET_PATH + OLD_DATA_PARQUET)
        )
        df_current = spark.read.parquet(_BUCKET_PATH + OLD_DATA_PARQUET)
    else:
        df_current = spark.createDataFrame(data=spark.sparkContext.emptyRDD(),
                                           schema=schema)
    
    df_new = spark.read.csv([_BUCKET_PATH + filename for filename in new_txts], schema=schema, header=True)

    # Converting into parquet for faster computing
    (
        df_new
        .repartition(N_PARTITIONS)
        .write 
        .mode("overwrite")
        .parquet(_BUCKET_PATH + NEW_DATA_PARQUET)
    )

    df_new = spark.read.parquet(_BUCKET_PATH + NEW_DATA_PARQUET)
    
    # Cleaning new dataset
    df_new = clean_dataset(df_new)
    
    # Concatenating current cleaned dataset with new cleaned data
    df_full = df_current.union(df_new)

    # Saving cleaned dataset
    (
        df_full
        .repartition(N_PARTITIONS)
        .write
        .mode("overwrite")
        .parquet(_BUCKET_PATH + CLEANED_DATASET_PARQUET)
    )
    
    spark.stop()

    # Saving and uploading new list of processed .txt files
    with open(PROCESSED_TXTS_PICKLE, "wb") as f:
        pickle.dump(all_txts, f)
    s3.upload_file(PROCESSED_TXTS_PICKLE, BUCKET_NAME, PROCESSED_TXTS_PICKLE)

    # Saving and uploading new data existence fact
    with open(NEW_DATA_PROCESSED_LOG, "w") as f:
            f.write("new data exists")
    s3.upload_file(NEW_DATA_PROCESSED_LOG, BUCKET_NAME, NEW_DATA_PROCESSED_LOG)

if __name__ == '__main__':
    main()
