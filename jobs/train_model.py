"""
This script trains a LightGBM model for house price prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated house age feature.
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.dbutils import DBUtils  # type: ignore
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from loan_prediction.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

dbutils = DBUtils(spark)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id

config_path = f"{root_path}/config/config.yml"
# config_path = ("/Volumes/mlops_test/loan_predictions/data/config/config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

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

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("person_income")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast YearBuilt to int for the function input
train_set = train_set.withColumn("YearBuilt", train_set["YearBuilt"].cast("int"))

# Feature engineering setup
training_set = fe.create_training_set(
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
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# # Calculate house_age for training and test set
# current_year = datetime.now().year
# test_set["house_age"] = current_year - test_set["YearBuilt"]

# Split features and target
X_train = training_df[num_features + cat_features]
y_train = training_df[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**parameters))])

mlflow.set_experiment(experiment_name="/Shared/loan-predictions-fe")

with mlflow.start_run(tags={"branch": "sweek5", "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
