import logging

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Simple yaml Config Loader from a file
    """

    def __init__(self, filepath="./config/config.yml"):
        """
            Init the loader with a file path for the project
        Args:
            filepath (str, optional): _description_. Defaults to './config/config.yml'.
        """
        self.filepath = filepath
        self.config = {}

    def load(self):
        """
        Public method to load config from yaml
        """

        # Load configuration
        with open(self.filepath, "r") as file:
            self.config = yaml.safe_load(file)

    def config_str(self):
        return yaml.dump(self.config, default_flow_style=False)


if __name__ == "__main__":
    config = ConfigLoader()
    config.load()
    logger.debug("Configuration loaded:")
    print(config.config_str())
