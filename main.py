import logging

from loan_prediction.config import ProjectConfig
from loan_prediction.data_loader import DataLoader
from loan_prediction.data_processor import DataProcessor

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load configuration
config = ProjectConfig.from_yaml(config_path="./config/config.yml")

logger.debug(config)
logger.info("Configuration loaded")

# Load data
filepath = "./data/sample/sample.csv"
# filepath = "./data/raw/train.csv"
dataloader = DataLoader(filepath)
dataloader.load()
logger.info(f"Data Loaded: {len(dataloader.data)}")


# Initialize DataProcessor
data_processor = DataProcessor(dataloader, config)
data_processor.load_data()
logger.info("DataProcessor initialized.")


logger.info(f"data_processor data shape: {data_processor.dataloader.data.shape}")


data_processor.preprocess()

logger.info("DataProcessor processed.")


# Split the data
# X_train, X_test, y_train, y_test = data_processor.split_train_test()
# logger.info("Data split into training and test sets.")
# logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

train_set, test_set = data_processor.split_train_test()


# run on databricks
# data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)


logger.info("Saved to catalog")

exit()

# # Initialize and train the model
# model = LoanClassifierModel(data_processor.preprocessor, configurator.config)
# model.train(X_train, y_train)
# logger.info("Model training completed.")


# # Evaluate the model
# score = model.evaluate(X_test, y_test)
# logger.info(f"Model evaluation completed: score={score}")


# ## Visualizing Results
# y_pred = model.predict(X_test)
# visualize_results(y_test, y_pred)
# logger.info("Results visualization completed.")

# ## Feature Importance
# feature_importance, feature_names = model.get_feature_importance()
# plot_feature_importance(feature_importance, feature_names)
# logger.info("Feature importance plot generated.")
