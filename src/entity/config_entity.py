from from_root import from_root
import os

class DatabaseConfig:
    """
    Configuration class for database settings.
    """
    def __init__(self):
        """
        Initialize DatabaseConfig with environment variables and default values.
        """
        self.USERNAME: str = os.environ["DATABASE_USERNAME"]
        self.PASSWORD: str = os.environ["DATABASE_PASSWORD"]
        self.URL: str = "mongodb+srv://<username>:<password>@projects.ch4mixt.mongodb.net/?retryWrites=true&w=majority"
        self.DBNAME: str = "ReverseImageSearchEngine"
        self.COLLECTION: str = "Embeddings"

    def get_database_config(self):
        """
        Get the database configuration as a dictionary.
        """
        return self.__dict__


class DataIngestionConfig:
    """
    Configuration class for data ingestion settings.
    """
    def __init__(self):
        """
        Initialize DataIngestionConfig with default values.
        """
        self.PREFIX: str = "images/"
        self.RAW: str = "data/raw"
        self.SPLIT: str = "data/splitted"
        self.BUCKET: str = "image-database-system-01"
        self.SEED: int = 1337
        self.RATIO: tuple = (0.8, 0.1, 0.1)

    def get_data_ingestion_config(self):
        """
        Get the data ingestion configuration as a dictionary.
        """
        return self.__dict__


class DataPreprocessingConfig:
    """
    Configuration class for data preprocessing settings.
    """
    def __init__(self):
        """
        Initialize DataPreprocessingConfig with default values.
        """
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train")
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test")
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "valid")

    def get_data_preprocessing_config(self):
        """
        Get the data preprocessing configuration as a dictionary.
        """
        return self.__dict__


class ModelConfig:
    """
    Configuration class for model settings.
    """
    def __init__(self):
        """
        Initialize ModelConfig with default values.
        """
        self.LABEL = 101
        self.STORE_PATH = os.path.join(from_root(), "model", "benchmark")
        self.REPOSITORY = 'pytorch/vision:v0.10.0'
        self.BASEMODEL = 'resnet18'
        self.PRETRAINED = True

    def get_model_config(self):
        """
        Get the model configuration as a dictionary.
        """
        return self.__dict__


class TrainerConfig:
    """
    Configuration class for training settings.
    """
    def __init__(self):
        """
        Initialize TrainerConfig with default values.
        """
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")
        self.EPOCHS = 2
        self.Evaluation = True

    def get_trainer_config(self):
        """
        Get the trainer configuration as a dictionary.
        """
        return self.__dict__


class ImageFolderConfig:
    """
    Configuration class for image folder settings.
    """
    def __init__(self):
        """
        Initialize ImageFolderConfig with default values.
        """
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images")
        self.IMAGE_SIZE = 256
        self.LABEL_MAP = {}
        self.BUCKET: str = "image-database-system-01"
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/images/{1}/{2}"

    def get_image_folder_config(self):
        """
        Get the image folder configuration as a dictionary.
        """
        return self.__dict__


class EmbeddingsConfig:
    """
    Configuration class for embeddings settings.
    """
    def __init__(self):
        """
        Initialize EmbeddingsConfig with default values.
        """
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")

    def get_embeddings_config(self):
        """
        Get the embeddings configuration as a dictionary.
        """
        return self.__dict__


class AnnoyConfig:
    """
    Configuration class for Annoy settings.
    """
    def __init__(self):
        """
        Initialize AnnoyConfig with default values.
        """
        self.EMBEDDING_STORE_PATH = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")

    def get_annoy_config(self):
        """
        Get the Annoy configuration as a dictionary.
        """
        return self.__dict__


class S3Config:
    """
    Configuration class for Amazon S3 settings.
    """
    def __init__(self):
        """
        Initialize S3Config with environment variables and default values.
        """
        self.ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
        self.S

