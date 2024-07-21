from datetime import datetime
import os

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, udf
from pyspark.sql.types import IntegerType


BUCKET = "s3a://mrozov-mlops/"
DATASET_NAME = "cleaned_dataset_sample.parquet"
RANDOM_STATE = 29
REGISTRY_SERVER_HOST = "10.0.0.3"
TRACKING_SERVER_HOST = "10.0.0.3"


def time_bin(hour):
    if 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    elif 18 <= hour < 24:
        return 3  # Evening
    else:
        return 4  # Night

def transform_data(df_cleaned: DataFrame) -> DataFrame:

    # Downsample the data, because there is a substantial imbalance in target column values
    df_majority = df_cleaned.filter(col("tx_fraud") == 0)
    df_minority = df_cleaned.filter(col("tx_fraud") == 1)

    df_majority_downsampled = df_majority.sample(withReplacement=False, fraction=df_minority.count() / df_majority.count())
    df_balanced = df_majority_downsampled.union(df_minority)

    # EDA showed obvious difference in target variable between samples with major value terminal_id=0 and others,
    # so it's a valuable feature
    # df.where(col("is_zero_terminal") == 0).agg({"tx_fraud": "avg"}).show() # this equals 0.0577
    # df.where(col("is_zero_terminal") == 1).agg({"tx_fraud": "avg"}).show() # this equals 0.0927
    transformed_df = df_balanced.withColumn("is_zero_terminal", when(col("terminal_id") == "0", 1).otherwise(0))

    # Using time from timestamp column in bins as a new attribute
    time_bin_udf = udf(time_bin, IntegerType())
    transformed_df = transformed_df.withColumn("time_of_day", time_bin_udf(hour(col("tx_datetime"))))

    return transformed_df

def get_pipeline() -> Pipeline:
    num_features = ["tx_amount", "tx_time_seconds", "is_zero_terminal"]
    cat_feature = "time_of_day"

    indexer = StringIndexer(inputCol=cat_feature, outputCol=cat_feature + "_index")
    oh_encoder = OneHotEncoder(inputCol=cat_feature + "_index", outputCol=cat_feature + "_encoded")

    all_features = num_features + [cat_feature + "_encoded"]

    assembler = VectorAssembler(inputCols=all_features, outputCol="features")
    scaler = StandardScaler(withMean=True, inputCol="features", outputCol="scaled_features")
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="tx_fraud")
    pipeline = Pipeline(stages=[indexer, oh_encoder, assembler, scaler, lr])

    return pipeline


def main():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_registry_uri(f"http://{REGISTRY_SERVER_HOST}:5000")
    spark = (
        SparkSession
        .builder
        .appName("Otus-hometask")
        .getOrCreate()
    )

    df = spark.read.parquet(BUCKET + DATASET_NAME)
    df = transform_data(df)

    experiment = mlflow.set_experiment("hometask_5_experiment")
    experiment_id = experiment.experiment_id
    run_name = f"run_{datetime.now()}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):

        pipeline = get_pipeline()
        lr = pipeline.getStages()[-1]

        train, test = df.randomSplit([0.8, 0.2], seed=RANDOM_STATE)

        param_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0, 0.1, 0.5])
            .addGrid(lr.maxIter, [100, 500, 1000])
            .build()
        )

        evaluator = BinaryClassificationEvaluator(labelCol="tx_fraud")
        
        tvs = TrainValidationSplit(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            parallelism=3,
            seed=RANDOM_STATE
        )

        model = tvs.fit(train)
        
        best_regParam = model.bestModel.stages[-1].getRegParam()
        best_maxIter = model.bestModel.stages[-1].getMaxIter()

        mlflow.log_param('optimal_regParam', best_regParam)
        mlflow.log_param('optimal_maxIter', best_maxIter)

        predictions = model.transform(test)
        roc_auc = evaluator.evaluate(predictions)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.spark.log_model(model, "hometask_5_model")
    
    spark.stop()

if __name__ == "__main__":
    main()
