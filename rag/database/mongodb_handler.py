import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from typing import Dict, List, Optional


class MongoDBHandler:
    """MongoDB is schema less, hence the responsibility of ensuring data consistency and integrity falls on the application logic.
    We perform document validation using schemas before they are inserted into MongoDB (done in 'ingestion/validation.py').
    """
    def __init__(self, connection_string, db_name):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # This will raise an exception if the connection fails
            self.db = self.client[self.db_name]
            self.logger.info(f"Connected to MongoDB. Database: {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to Mongodb: {e}")
            raise

    def close_connection(self):
        if self.client:
            self.client.close()
            self.logger.info("Connection to MongoDB closed.")

    def create_collection(self, collection_name, **kwargs):
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name, **kwargs)
            self.logger.info(f"Collection '{collection_name}' created.")
        else:
            self.logger.warn(f"Collection '{collection_name}' already exists.")

    def upsert_document(self, collection_name, filter_dict, update_dict):
        """ 
        Insert a new document if no document matches the filter, otherwise update the existing document.
        filterd_dict: A dictionary containing the filter criteria to find the document.
        update_dict: A dictionary containing the fields and values to be updated in the document.
        """
        collection = self.db[collection_name]
        result = collection.update_one(filter_dict, {"$set": update_dict}, upsert=True)
        return result
    
    def bulk_upsert(self, collection_name: str, documents: List[Dict], id_field: str) -> Optional[Dict[str, int]]:
        """
        Perform bulk upsert operation on documents in the collection.
        Note: We use custom id field for the document, namely act_id, case_id, etc. which are different than 
                the default _id field used by MongoDB.

        All documents must contain the `id_field`. The `id_field` is used to match existing documents
        and no default ObjectId will be generated for new documents.

        Parameters:
        - collection_name: Name of the MongoDB collection.
        - documents: List of dictionaries where each dictionary represents a document.
        - id_field: Field used to uniquely identify documents.

        Returns:
        - A dictionary with counts of inserted and modified documents, or None in case of an error.
        e.g. {'inserted': 10, 'modified': 5, 'total': 15}
        """
        collection = self.db[collection_name]
        operations = []
        for doc in documents:
            if id_field not in doc:
                raise ValueError(f"Document is missing the id field '{id_field}'")
            
            filter_dict = {id_field: doc[id_field]}
            operations.append({
                'update_one': {
                    'filter': filter_dict,
                    'update': {'$set': doc},
                    'upsert': True
                }
            })

        try:
            result = collection.bulk_write(operations)
            return {
                'inserted': result.upserted_count,
                'modified': result.modified_count,
                'total': len(documents)
            }
        except Exception as bwe:
            self.logger.error(f"Bulk write error: {bwe}")
            return None
        
    def find_documents(self, collection_name, filter_dict=None, projection=None, limit=0, skipi=0):
        """
        Find documents in the collection that match the filter criteria.
        """
        if filter_dict is None:
            filter_dict = {}
        if projection is None:
            projection = {}

        collection = self.db[collection_name]
        return list(collection.find(filter_dict, projection).skip(skipi).limit(limit))
    
    def count_documents(self, collection_name, filter_dict=None):
        """
        Count documents in the collection that match the filter criteria.
        """
        # Ensure filter_dict is a dictionary, defaulting to an empty dictionary if None
        if filter_dict is None:
            filter_dict = {}

        collection = self.db[collection_name]
        return collection.count_documents(filter_dict)

    def delete_documents(self, collection_name, filter_dict):
        """
        Delete documents from the collection that match the filter criteria.
        """
        collection = self.db[collection_name]
        result = collection.delete_many(filter_dict)
        return result.deleted_count

    def create_index(self, collection_name, keys, **kwargs):
        """
        Create an index on the specified collection.
        Note: the callers 

        :param collection_name: Name of the collection
        :param keys: A single key or a list of (key, direction) pairs
        :param kwargs: Additional arguments to pass to create_index
        :return: The name of the created index
        """                
        collection = self.db[collection_name]
        index_name = collection.create_index(keys, **kwargs)
        print(f"Index '{index_name}' created on collection '{collection_name}'.")
        return index_name

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_connection()
