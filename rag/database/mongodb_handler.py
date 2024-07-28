import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

class MongoDBHandler:
    def __init__(self, connection_string, db_name):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        

    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # This will raise an exception if the connection fails
            self.db = self.client[self.db_name]
            print(f"Connected to MongoDB. Database: {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"Failed to connect to Mongodb: {e}")
            raise

    def create_collection(self, collection_name, **kwargs):
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name, **kwargs)
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def create_index(self, collection_name, keys, **kwargs):
        collection = self.db[collection_name]
        index_name = collection.create_index(keys, **kwargs)
        print(f"Index '{index_name}' created on collection '{collection_name}'.")

    def insert_one(self, collection_name, document):
        collection = self.db[collection_name]
        collection.insert_one(document)
        print(f"Document inserted into collection '{collection_name}'.")

    def insert_many(self, collection_name, documents):
        collection = self.db[collection_name]
        collection.insert_many(documents)
        print(f"Documents inserted into collection '{collection_name}'.")

    def find_one(self, collection_name, query, projection=None):
        collection = self.db[collection_name]
        return collection.find_one(query, projection)
    
    def find(self, collection_name, query, projection=None, limit=0, skip=0):
        collection = self.db[collection_name]
        return collection.find(query, projection, limit=limit, skip=skip)

    def update_one(self, collection_name, filter, update):
        collection = self.db[collection_name]
        result = collection.update_one(filter, update)
        return result.modified_count
    
    def update_many(self, collection_name, filter, update):
        collection = self.db[collection_name]
        result = collection.update_many(filter, update)
        return result.modified_count
    
    def delete_one(self, collection_name, filter):
        collection = self.db[collection_name]
        result = collection.delete_one(filter)
        return result.deleted_count
    
    def count_documents(self, collection_name, filter):
        collection = self.db[collection_name]
        return collection.count_documents(filter or {})
    
    def close(self):
        if self.client:
            self.client.close()
            print("Connection to MongoDB closed.")