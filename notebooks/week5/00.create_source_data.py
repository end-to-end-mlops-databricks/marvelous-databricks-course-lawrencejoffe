# Databricks notebook source
# MAGIC %pip install ../mlops_with_databricks-0.0.1.tar.gz

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from loan_prediction.config import ProjectConfig



# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="/Volumes/heiaepgah71pwedmld01001/lj_loan_prediction/packages/config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

print(f"{catalog_name=}")
print(f"{schema_name=}")

# COMMAND ----------

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)

combined_set.head(5).T


# COMMAND ----------

existing_ids = set(int(id) for id in combined_set['id'])

# COMMAND ----------

# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100):
    synthetic_data = pd.DataFrame()
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != 'id':
            if column in ['person_age']:
                synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)  # Years between existing values
            else:
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)
        
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
            
        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, num_rows)
                )
            else:
                synthetic_data[column] = [min_date] * num_rows
        
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)
    
    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data['id'] = new_ids

    return synthetic_data



# COMMAND ----------

# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

#comment out on first time run so that table is created

existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data").schema
synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

#first time run - uncomment
#synthetic_spark_df = spark.createDataFrame(synthetic_df)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

train_set_with_timestamp.describe()

# COMMAND ----------

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.source_data"
)

# COMMAND ----------

#first time when table is created
# spark.sql(
#             f"ALTER TABLE {catalog_name}.{schema_name}.source_data "
#             "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
#         )

# COMMAND ----------

display(spark.sql(f"select count(*) from {catalog_name}.{schema_name}.source_data"))