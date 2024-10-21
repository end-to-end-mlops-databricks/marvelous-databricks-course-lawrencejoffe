import logging

from loan_prediction.config_loader import ConfigLoader
from loan_prediction.data_loader import DataLoader
from loan_prediction.data_processor import DataProcessor
from loan_prediction.loan_classifier_model import LoanClassifierModel
from loan_prediction.utils import plot_feature_importance, visualize_results

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load configuration
configurator = ConfigLoader()
configurator.load()
logger.debug(configurator.config_str())
logger.info("Configuration loaded")


# Load data
filepath = "./data/sample/sample.csv"
filepath = "./data/raw/train.csv"
dataloader = DataLoader(filepath)
dataloader.load()
logger.info("Data Loaded")


# Initialize DataProcessor
data_processor = DataProcessor(dataloader, configurator.config)
data_processor.load_data(filepath)
logger.info("DataProcessor initialized.")

logger.info(f"data_processor data size: {data_processor.df.size}")

data_processor.process()

logger.info("DataProcessor built.")


# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Initialize and train the model
model = LoanClassifierModel(data_processor.preprocessor, configurator.config)
model.train(X_train, y_train)
logger.info("Model training completed.")


# Evaluate the model
score = model.evaluate(X_test, y_test)
logger.info(f"Model evaluation completed: score={score}")


## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)
logger.info("Results visualization completed.")

## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
logger.info("Feature importance plot generated.")
