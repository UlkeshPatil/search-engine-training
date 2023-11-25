from src.entity.config_entity import DataIngestionConfig
from src.utils.storage_handler import S3Connector
from from_root import from_root
import splitfolders
import os


class DataIngestion:
    """
    Class for ingesting and splitting data for the image database system.
    """
    def __init__(self):
        """
        Initialize DataIngestion with configuration settings.
        """
        self.config = DataIngestionConfig()

    def download_dir(self):
        """
        Download data from S3 to the local directory.

        Raises:
            Exception: If an error occurs during the download process.
        """
        try:
            print("\n====================== Fetching Data ==============================\n")
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            os.system(f"aws s3 sync s3://image-database-system-01/images/ {data_path} --no-progress")
            print("\n====================== Fetching Completed ==========================\n")

        except Exception as e:
            raise e

    def split_data(self):
        """
        Split data into train, validation, and test sets.

        Raises:
            Exception: If an error occurs during the data splitting process.
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise e

    def run_step(self):
        """
        Run the data ingestion process, including downloading and splitting.

        Returns:
            dict: A response dictionary indicating the completion of the data ingestion process.
        """
        self.download_dir()
        self.split_data()
        return {"Response": "Completed Data Ingestion"}


if __name__ == "__main__":
    paths = ["data", r"data\raw", r"data\splitted", r"data\embeddings",
             "model", r"model\benchmark", r"model\finetuned"]

    for folder in paths:
        path = os.path.join(from_root(), folder)
        print(path)
        if not os.path.exists(path):
            os.mkdir(folder)

    dc = DataIngestion()
    print(dc.run_step())

