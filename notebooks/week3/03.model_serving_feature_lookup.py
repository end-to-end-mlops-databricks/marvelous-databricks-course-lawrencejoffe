# Databricks notebook source
# MAGIC %pip install ../mlops_with_databricks-0.0.1.tar.gz

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Table for loan features
# MAGIC We already created loan_features table as feature look up table.

# COMMAND ----------

import time

import requests
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

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../config/config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.loan_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.loan_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

from databricks.sdk.service.catalog import *

online_table = OnlineTable(
  name=online_table_name,
  spec=spec
)

online_table_pipeline = workspace.online_tables.create_and_wait(table=online_table)

# COMMAND ----------


# config = ProjectConfig.from_yaml(config_path="/Volumes/mlops_dev/house_prices/data/project_config.yml")

# catalog_name = config.catalog_name
# schema_name = config.schema_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create endpoint

# COMMAND ----------

workspace.serving_endpoints.create(
    name="lj-loan-prediction-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.lj_loan_prediction_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

num_features = config.num_features
cat_features = config.cat_features
required_columns = cat_features + num_features
required_columns

# COMMAND ----------

fe_columns = ["person_income"]
for f in fe_columns:
    required_columns.remove(f)

required_columns  

# COMMAND ----------



train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

train_set.dtypes

# COMMAND ----------

dataframe_records[0]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/lj-loan-prediction-model-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

loan_features = spark.table(f"{catalog_name}.{schema_name}.loan_features").toPandas()

# COMMAND ----------

loan_features.dtypes