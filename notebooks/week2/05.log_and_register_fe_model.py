# Databricks notebook source
# MAGIC %pip install ../housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython() 

# COMMAND ----------

import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import functions as F
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from loan_prediction.config import ProjectConfig
import logging

logger = logging.getLogger(__name__)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------



# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml(config_path="../../config/config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.loan_features"
function_name = f"{catalog_name}.{schema_name}.calculate_person_age_months"


# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


# COMMAND ----------

# Create or replace the loan_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.loan_features
(
    Id STRING NOT NULL,
    credit_history_length INT,
    loan_amount INT,
    person_age INT,
    person_income INT
 );
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.loan_features "
          "ADD CONSTRAINT loan_pk PRIMARY KEY(Id);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.loan_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")


# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.loan_features "
          f"SELECT id, cb_person_cred_hist_length, loan_amnt, person_age,person_income FROM {catalog_name}.{schema_name}.train_set")
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.loan_features "
          f"SELECT id, cb_person_cred_hist_length, loan_amnt, person_age,person_income FROM {catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Define a function to calculate the perons's age in months
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(person_age INT)
RETURNS INT
LANGUAGE PYTHON AS
$$
from datetime import datetime
return 12 * person_age
$$
""")

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")



# COMMAND ----------

# Load training and test sets
train_set = (
        spark.table(f"{catalog_name}.{schema_name}.train_set")
            .drop(
                # "cb_person_cred_hist_length", 
                #   "loan_amnt" , 
                  "person_income")
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast person_age to int for the function input
train_set = train_set.withColumn("person_age", train_set["person_age"].cast("int"))
train_set = train_set.withColumn("id", train_set["id"].cast("string"))

# Feature engineering setup
training_set_fe = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["credit_history_length", "loan_amount", "person_income"],
            lookup_key="id",
        ),

        FeatureFunction(
            udf_name=function_name,
            output_name="person_age_months",
            input_bindings={"person_age": "person_age"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)



# COMMAND ----------

parameters

# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set_fe.load_df().toPandas()

# Split features and target
X_train = training_df[num_features + cat_features]
y_train = training_df[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestClassifier(
        n_estimators = parameters['n_estimators'], max_depth = parameters['max_depth'], random_state=42))]
)


# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/lj_loan_prediction-fe")
git_sha = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R2 Score: {r2}")

    accuracy = pipeline.score(X_test, y_test)
    print(f"Score: {accuracy}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForestClassifier with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("score", accuracy)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="randomforest-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f'runs:/{run_id}/randomforest-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.lj_loan_prediction_model_fe")
    


