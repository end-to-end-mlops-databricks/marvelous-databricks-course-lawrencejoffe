import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, dataloader: DataLoader, config):
        self.dataloader = dataloader
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None
        self.df = None

    def load_data(self):
        self.dataloader.load()
        # data sitting in self.dataloader.data

    def process(self):
        # Remove rows with missing target
        target = self.config.target
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config.num_features + self.config.cat_features]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""

        self.df = self.dataloader.data

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # example for filing missing values
        # median_age = self.df['person_age'].median()
        # self.df['person_age'].fillna(median_age, inplace=True)

        # current_year = datetime.now().year
        # self.df['person_age'] = current_year - self.df['person_birth']
        # self.df.drop(columns=['person_birth'], inplace=True)

        # Fill missing values with mean or default values
        # self.df.fillna({
        #     'LotFrontage': self.df['LotFrontage'].mean(),
        #     'MasVnrType': 'None',
        #     'MasVnrArea': 0,
        # }, inplace=True)

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        # TODO make sure there is an 'id' column in the data
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + ["id"]

        self.df = self.df[relevant_columns]

    # def split_data(self, test_size=0.2, random_state=42):
    #     return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def split_train_test(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        logger.info(f"Saving to catalog {self.config.catalog_name}.{self.config.schema_name}")

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
