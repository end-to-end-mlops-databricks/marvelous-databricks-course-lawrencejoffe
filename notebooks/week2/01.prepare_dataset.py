# Databricks notebook source
# !pip install /Volumes/heiaepgah71pwedmld01001/lj_loan_prediction/packages/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from loan_prediction.data_processor import DataProcessor
from loan_prediction.config import ProjectConfig
from loan_prediction.data_loader import DataLoader
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../config/config.yml")

# COMMAND ----------

#Load data
filepath = '/Volumes/heiaepgah71pwedmld01001/lj_loan_prediction/data/train.csv'
dataloader = DataLoader(filepath)
dataloader.load()
logger.info(f"Data Loaded: {len(dataloader.data)}")



# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(dataloader, config)
data_processor.load_data()
logger.info("DataProcessor initialized.")


data_processor.preprocess()
logger.info("DataProcessor processed.")

train_set, test_set = data_processor.split_train_test()



# COMMAND ----------


data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
logger.info("Saved to catalog")


# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from heiaepgah71pwedmld01001.lj_loan_prediction.train_set
