from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import AnnoyConfig
from annoy import AnnoyIndex
from typing_extensions import Literal
from tqdm import tqdm
import json


class CustomAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        """
        Custom implementation of AnnoyIndex with support for storing labels.

        Parameters:
        - f (int): The number of dimensions in the vector.
        - metric (Literal["angular", "euclidean", "manhattan", "hamming", "dot"]): Distance metric.

        """
        super().__init__(f, metric)
        self.label = []

    def add_item(self, i: int, vector, label: str) -> None:
        """
        Add item to the index with a corresponding label.

        Parameters:
        - i (int): Item index.
        - vector: Item vector.
        - label (str): Item label.

        """
        super().add_item(i, vector)
        self.label.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        """
        Get the nearest neighbors by vector.

        Parameters:
        - vector: Query vector.
        - n (int): Number of neighbors to retrieve.
        - search_k (int): Search parameter.
        - include_distances (Literal[False]): Whether to include distances in the result.

        Returns:
        - List[str]: Labels of the nearest neighbors.

        """
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels = [self.label[link] for link in indexes]
        return labels

    def load(self, fn: str, prefault: bool = ...):
        """
        Load the index and corresponding labels from files.

        Parameters:
        - fn (str): File name of the index.
        - prefault (bool): Whether to prefault.

        """
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...):
        """
        Save the index and corresponding labels to files.

        Parameters:
        - fn (str): File name of the index.
        - prefault (bool): Whether to prefault.

        """
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Annoy(object):
    def __init__(self):
        """
        Annoy class for building and managing Annoy index.

        """
        self.config = AnnoyConfig()
        self.mongo = MongoDBClient()
        self.result = self.mongo.get_collection_documents()["Info"]

    def build_annoy_format(self):
        """
        Build Annoy index and store it in the specified file.

        Returns:
        - bool: True if successful.

        """
        Ann = CustomAnnoy(256, 'euclidean')
        print("Creating Ann for predictions : ")
        for i, record in tqdm(enumerate(self.result), total=8677):
            Ann.add_item(i, record["images"], record["s3_link"])

        Ann.build(100)
        Ann.save(self.config.EMBEDDING_STORE_PATH)
        return True

    def run_step(self):
        """
        Run the Annoy building step.

        """
        self.build_annoy_format()


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()
