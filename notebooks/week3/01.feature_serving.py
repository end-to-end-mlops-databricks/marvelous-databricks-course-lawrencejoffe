# Databricks notebook source
# MAGIC %pip install ../mlops_with_databricks-0.0.1.tar.gz

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from loan_prediction.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config, train and test tables

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../config/config.yml")

# Get feature columns details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.loan_prediction_preds"
online_table_name = f"{catalog_name}.{schema_name}.loan_prediction_preds_online_table"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

df.head(5).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.pyfunc.load_model(f"models:/{catalog_name}.{schema_name}.loan_prediction_model_pyfunc/2")

# pipeline = mlflow.pyfunc.load_model(f"models:/{catalog_name}.{schema_name}.lj_loan_prediction_model_fe/1")

# COMMAND ----------

df.columns

# COMMAND ----------

feature_columns = ["id","loan_percent_income"]

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[feature_columns]
preds_df["predicted_loan_status"] = pipeline.predict(df[cat_features + num_features])

preds_df = spark.createDataFrame(preds_df)


# COMMAND ----------

preds_df

# COMMAND ----------

feature_table_name

# COMMAND ----------


# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, 
    primary_keys=["id"], 
    df=preds_df, 
    description="Loan status predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

online_table_name

# COMMAND ----------

from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
import mlflow

# COMMAND ----------

# breaking changes to API in databricks-sdk > 0.20. Refer to https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html for new syntax for create and wait

# COMMAND ----------

# 2. Create the online table using feature table

spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False
)

online_table = OnlineTable(
  name=online_table_name,
  spec=spec
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create_and_wait(table=online_table)



# COMMAND ----------

## explanation for below creating a feature spec
# ### https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html
#When you create a feature spec, you specify the source Delta table. This allows the feature spec to be used in #both offline and online scenarios. For online lookups, the serving endpoint automatically uses the online #table to perform low-latency feature lookups.



# COMMAND ----------

#fe.delete_feature_spec(name=feature_spec_name)

# COMMAND ----------

# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name, lookup_key="id", feature_names=["loan_percent_income", "predicted_loan_status"]
    )
]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions_feature_spec"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------

# 4. Create endpoing using feature spec

# Create a serving endpoint for the house prices predictions


endpoint_name = 'lj-loan-prediction-feature-serving'

#updated api
workspace.serving_endpoints.create_and_wait( 
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)




# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint

# COMMAND ----------



# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

id_list = preds_df["id"]

# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/lj-loan-prediction-feature-serving/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"id": "10"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------

# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["id"], "data": [["10"]]}},
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/lj-loan-prediction-feature-serving/invocations"
id_list = preds_df.select("id").rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10


# Function to make a request and record latency
def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")