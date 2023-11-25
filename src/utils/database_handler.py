from src.entity.config_entity import DatabaseConfig
from pymongo import MongoClient
from typing import List, Dict, Any


class MongoDBClient:
    """
    Class for interacting with MongoDB.
    """
    def __init__(self):
        """
        Initialize MongoDBClient with configuration settings and establish a connection to MongoDB.
        """
        self.config = DatabaseConfig()
        url = self.config.URL.replace("<username>", self.config.USERNAME).replace("<password>", self.config.PASSWORD)
        self.client = MongoClient(url)

    def insert_bulk_record(self, documents: List[Dict[str, Any]]):
        """
        Insert a list of documents into the MongoDB collection.

        Args:
            documents (List[Dict[str, Any]]): List of documents to be inserted.

        Returns:
            dict: A response dictionary indicating the success and the number of inserted documents.
        """
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            result = db[collection].insert_many(documents)
            return {"Response": "Success", "Inserted Documents": len(result.inserted_ids)}
        except Exception as e:
            raise e

    def get_collection_documents(self):
        """
        Retrieve all documents from the MongoDB collection.

        Returns:
            dict: A response dictionary indicating the success and the retrieved documents.
        """
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            result = db[collection].find()
            return {"Response": "Success", "Info": result}
        except Exception as e:
            raise e

    def drop_collection(self):
        """
        Drop the MongoDB collection.

        Returns:
            dict: A response dictionary indicating the success of dropping the collection.
        """
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            db[collection].drop()
            return {"Response": "Success"}
        except Exception as e:
            raise e


if __name__ == "__main__":
    data = [
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"}
    ]

    mongo = MongoDBClient()
    print(mongo.insert_bulk_record(data))
    # Uncomment the following lines to test the methods
    # print(mongo.drop_collection())
    # result = mongo.get_collection_documents()
    # print(result["Info"])

