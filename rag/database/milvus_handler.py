from pymilvus import DataType, CollectionSchema, FieldSchema, Collection, connections, utility, MilvusClient
from pymilvus.exceptions import SchemaNotReadyException
from config.config import ConfigLoader
from utils.logger.logging_config import logger
from typing import Dict, Any

class MilvusHandler:
    def __init__(self):
        self.logger = logger
        self.config = ConfigLoader().get_milvus_config()
        self.client = None
        self.default_field = "embedding"  # Default field for vector embeddings
        self.schemas = self._initialize_schemas()
        self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_connection()
    
    def get_collection_config(self, collection_name: str):
        return self.config.collections.get(collection_name)

    def get_index_params(self, collection_name:str):
        collection_config = self.get_collection_config(collection_name)
        if collection_config:
            return collection_config.index_params
        raise ValueError(f"Configuration for collection '{collection_name}' not found")
        
    # will be used in case search_params are not provided by user of search function
    def get_default_search_params(self, collection_name:str):
        collection_config = self.get_collection_config(collection_name)
        if collection_config:
            return collection_config.search_params
        raise ValueError(f"Configuration for collection '{collection_name}' not found")

    def connect(self):
        # Connect to Milvus server
        self.logger.info(f"Connecting to Milvus at {self.config.host}:{self.config.port}")
        try:
            connections.connect(
                alias="default", 
                host=self.config.host, 
                port=self.config.port
            )
            self.client = MilvusClient(uri=f"http://{self.config.host}:{self.config.port}")
            self.logger.info(f"Connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            self.client = None

    def close_connection(self):
        # Close connection to Milvus server
        if self.client:
            self.client.close()
            self.logger.info("Connection to Milvus closed.")

    def _initialize_schemas(self):
        # Define schemas for all collections
        """
        Defines how the vector data (e.g. embeddings) and associated metadata should be stored in Milvus.

        Primary Key Uniqueness:
            If a document with the same primary key is inserted multiple times, the latest version will overwrite the previous one.
            Insert actually becomes upsert in this case.
        """
        return {
            "legal_acts": [
                FieldSchema(name="act_id", dtype=DataType.INT64, max_length=100, is_primary=True, description="Unique identifier for the act"),
                FieldSchema(name="chunk_id", dtype=DataType.INT64, description="Unique identifier for the chunk"),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768, description="Text content embeddings"),
                FieldSchema(name="act_year", dtype=DataType.VARCHAR, max_length=30, description="Year of the act"),
                FieldSchema(name="short_title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="long_title", dtype=DataType.VARCHAR, max_length=1000),
            ],
            "case_files": [
                FieldSchema(name="case_id", dtype=DataType.INT64, max_length=100, is_primary=True, description="Unique identifier for the case"),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768, description="Text content embeddings"),
                FieldSchema(name="case_year", dtype=DataType.INT64),
                FieldSchema(name="case_title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="plaintiff", dtype=DataType.VARCHAR, max_length=300),
                FieldSchema(name="defendant", dtype=DataType.VARCHAR, max_length=300),
            ],
            # Add more schemas here as needed
        }
    
    # Return the schema for a given collection
    def get_schema(self, collection_name):
        if collection_name in self.schemas:
            return self.schemas[collection_name]
        else:
            self.logger.error(f"Schema for collection '{collection_name}' not found.")
            return None

    # function to create a collection with a given name
    def create_collection(self, collection_name, force=False):
        if self.client is None:
            self.logger.critical("Milvus client is not connected. Please connect first.")
            raise RuntimeError("Milvus client is not connected. Please connect first.")
        
        # Drop the collection if it already exists and force=True
        if force and utility.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            self.logger.info(f"Collection '{collection_name}' dropped.")
        else:
            self.logger.warn(f"Collection '{collection_name}' does not exist, cannot drop")

        #Create collection if it does not exist
        if not utility.has_collection(collection_name):
            if collection_name in self.schemas:
                # Create schema for the collection
                schema = CollectionSchema(fields=self.schemas[collection_name], description=f"Schema for {collection_name}")
                schema.verify()
                # Create collection with the given schema
                self.client.create_collection(collection_name=collection_name, schema=schema)
                # Create index after creating collection
                self.create_index(collection_name=collection_name)
                self.logger.info(f"Collection '{collection_name}' created with schema: {schema}")
            else:
                self.logger.error(f"Schema for collection '{collection_name}' not found.")
        else:
            self.logger.warn(f"Collection '{collection_name}' already exists, skipping creation.")

    def validate_search_params(self, collection_name: str, search_params: Dict[str, Any]) -> Dict[str, Any]:
        index_params = self.get_index_params(collection_name)
        default_search_params = self.get_default_search_params(collection_name)

        # Ensure metric_type matches index_params
        if 'metric_type' in search_params and search_params['metric_type'] != index_params['metric_type']:
            self.logger.error(f"Search metric_type must match index metric_type: {index_params['metric_type']}. Fix to proceed.")
            raise ValueError(f"Search metric_type must match index metric_type: {index_params['metric_type']}")
        
        # Set default metric_type if not provided
        if 'metric_type' not in search_params:
            search_params['metric_type'] = index_params['metric_type']
        
        # Validate and set default params based on index_type
        if index_params['index_type'] == 'IVF_FLAT':
            if 'nprobe' not in search_params:
                search_params['nprobe'] = default_search_params['params'].get('nprobe', 10)
            else:
                # ensure nprobe is within valid range (1, nlist)
                search_params['nprobe'] = max(1, min(search_params['nprobe'], index_params['params']['nlist']))
        elif index_params['index_type'] == 'HNSW':
            if 'ef' not in search_params:
                search_params['ef'] = default_search_params['params'].get('ef', 10)
        
        return search_params

    def load_collection(self, collection_name):
        self.client.load_collection(collection_name)
        # TODO: Add error handling, check load state (client.get_load_state()); ref https://milvus.io/docs/manage-collections.md#Load--Release-Collection

    def get_collection(self, collection_name):
        try:
            return Collection(collection_name, using="default")
        except SchemaNotReadyException as e:
            self.logger.error(f"Collection '{collection_name}' does not exist or schema is not ready.")
            return None

    def create_index(self, collection_name):
        collection = Collection(name=collection_name, using="default")
        if not collection.has_index():
            collection.create_index(field_name="embedding", index_params=self.get_index_params(collection_name))

    def insert(self, vectors, collection_name):
        # Ensure collection exists
        if not self.client.has_collection(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create it first.")
            return
        # Get the collection
        collection = Collection(collection_name, using="default")
        # Insert vectors into the specified collection
        collection.insert(vectors)
        # Automatic flushing by Milvus, no need to call flush explicitly
        # Ensure index is created (index is managed internally)
        self.create_index(collection_name)

    def describe_collection(self, collection_name):
        if utility.has_collection(collection_name, using="default"):
            return Collection(collection_name, using="default").describe()
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            return None
  
    def search(self, collection_name, query_vectors, top_k, field_name=None, search_params=None, output_fields=None):
        # Ensure the collection exists
        if not utility.has_collection(collection_name, using="default"):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create it first.")
            return
        
        # Get the collection
        collection = Collection(collection_name, using="default")
        collection.load()

        # Use provided field_name if available, else use default
        field_name = field_name if field_name else self.default_field

        # Use provided search_params if available, else use default
        if search_params is None:
            search_params = self.get_default_search_params(collection_name)
        else:
            search_params = self.validate_search_params(collection_name, search_params)

        # Perform the search
        #results = collection.search(query_vectors, field_name, search_params, top_k, output_fields=output_fields)
        results = self.client.search(
            collection_name=collection_name, 
            data=query_vectors, 
            anns_field=field_name, 
            params=search_params, 
            limit=top_k, 
            output_fields=output_fields
        )
        
        return results

    def delete(self, collection_name):
        if utility.has_collection(collection_name, using="default"):
            collection = Collection(collection_name, using="default")
            collection.drop()
            self.logger.info(f"Collection '{collection_name}' dropped.")
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")

    def count(self, collection_name):
        if utility.has_collection(collection_name, using="default"):
            collection = Collection(collection_name, using="default")
            return collection.num_entities
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            return None
        
    def list_collections(self):
        # List all collections
        return utility.list_collections(using="default")
    
    def drop_index(self, collection_name):
        if utility.has_collection(collection_name, using="default"):
            collection = Collection(name=collection_name, using="default")
            if collection.has_index():
                collection.drop_index()
                self.logger.info(f"Index for collection '{collection_name}' has been dropped.")
            else:
                self.logger.error(f"Collection '{collection_name}' does not have an index.")
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")

    def update_collection_schema(self, collection_name, new_schema):
        if utility.has_collection(collection_name, using="default"):
            collection = Collection(name=collection_name, using="default")
            collection.drop()
            self.client.create_collection(collection_name, schema=new_schema)
            self.logger.info(f"Collection '{collection_name}' schema has been updated.")
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")

    def check_index_status(self, collection_name):
        if utility.has_collection(collection_name, using="default"):
            collection = Collection(name=collection_name, using="default")
            if collection.has_index():
                self.logger.info(f"Collection '{collection_name}' has an index.")
            else:
                self.logger.error(f"Collection '{collection_name}' does not have an index.")
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")

    def define_default_schema(self, collection_name):
        # Define a default schema for testing purposes
        if collection_name not in self.schemas:
            # Example schema: id (primary key), embedding (vector field)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  # Adjust dimension as needed
            ]
            self.schemas[collection_name] = CollectionSchema(fields, description="Default schema for testing.")