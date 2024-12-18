# Databricks notebook source

!pip install /Volumes/heiaepgah71pwedmld01001/lj_loan_prediction/packages/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------
from pyspark.sql import SparkSession
from loan_prediction.config import ProjectConfig
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc') # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../config/config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------
# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], 
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(**parameters))
])


# COMMAND ----------
mlflow.set_experiment(experiment_name='/Shared/lj_loan_prediction')
git_sha = "ffa63b430205ff7"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(
    train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm-pipeline-model",
        signature=signature
    )


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model',
    name=f"{catalog_name}.{schema_name}.loan_prediction_basic",
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------

# from the endpoint suggested test 

from mlflow.models import validate_serving_input

model_uri = 'runs:/933b545bd39e48e78738ad4ba649d7c1/lightgbm-pipeline-model'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(X_test)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------

