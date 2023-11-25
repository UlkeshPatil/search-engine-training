from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.embeddings import EmbeddingGenerator, ImageFolder
from src.utils.storage_handler import S3Connector
from src.components.nearest_neighbours import Annoy
from src.components.model import NeuralNet
from src.components.trainer import Trainer
from torch.utils.data import DataLoader
from from_root import from_root
from tqdm import tqdm
import torch
import os


class Pipeline:
    def __init__(self):
        """
        Initialize the image search pipeline.
        """
        self.paths = ["data", "data/raw", "data/splitted", "data/embeddings",
                      "model", "model/benchmark", "model/finetuned"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initiate_data_ingestion(self):
        """
        Initialize and run the data ingestion process.
        """
        for folder in self.paths:
            path = os.path.join(from_root(), folder)
            if not os.path.exists(path):
                os.mkdir(folder)

        dc = DataIngestion()
        dc.run_step()

    @staticmethod
    def initiate_data_preprocessing():
        """
        Initialize and run the data preprocessing process.

        Returns:
            dict: Data loaders for training, testing, and validation.
        """
        dp = DataPreprocessing()
        loaders = dp.run_step()
        return loaders

    @staticmethod
    def initiate_model_architecture():
        """
        Initialize the neural network model.

        Returns:
            NeuralNet: Instance of the neural network model.
        """
        return NeuralNet()

    def initiate_model_training(self, loaders, net):
        """
        Initialize and run the model training process.

        Args:
            loaders (dict): Data loaders for training, testing, and validation.
            net (NeuralNet): Instance of the neural network model.
        """
        trainer = Trainer(loaders, self.device, net)
        trainer.train_model()
        trainer.evaluate(validate=True)
        trainer.save_model_in_pth()

    def generate_embeddings(self, loaders, net):
        """
        Generate embeddings using the trained model.

        Args:
            loaders (dict): Data loaders for training, testing, and validation.
            net (NeuralNet): Instance of the neural network model.
        """
        data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
        dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
        embeds = EmbeddingGenerator(model=net, device=self.device)

        for batch, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            print(embeds.run_step(batch, img, target, link))

    @staticmethod
    def create_annoy():
        """
        Create Annoy index using generated embeddings.
        """
        ann = Annoy()
        ann.run_step()

    @staticmethod
    def push_artifacts():
        """
        Push artifacts to storage (S3).
        """
        connection = S3Connector()
        response = connection.zip_files()
        return response

    def run_pipeline(self):
        """
        Run the complete image search pipeline.

        Returns:
            dict: Response indicating the completion of the pipeline.
        """
        self.initiate_data_ingestion()
        loaders = self.initiate_data_preprocessing()
        net = self.initiate_model_architecture()
        self.initiate_model_training(loaders, net)
        self.generate_embeddings(loaders, net)
        self.create_annoy()
        self.push_artifacts()
        return {"Response": "Pipeline Run Complete"}


if __name__ == "__main__":
    image_search = Pipeline()
    image_search.run_pipeline()
