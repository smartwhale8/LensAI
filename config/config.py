import json
from pathlib import Path
from pydantic import BaseModel, Field


import json
from typing import Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

class MongoDBConfig(BaseModel):
    host: str
    port: int
    database: str

class MilvusCollectionConfig(BaseModel):
    collection_name: str
    index_params: Dict[str, Any]
    search_params: Dict[str, Any]

class MilvusConfig(BaseModel):
    host: str
    port: int
    collections: Dict[str, MilvusCollectionConfig]

class EmbeddingConfig(BaseModel):
    emb_model_name: str

class GeneratorConfig(BaseModel):
    gen_model_name: str
    gen_model_file: str
    gen_model_type: str
    gen_tokenizer: str
    gpu_layers: int

class Config(BaseModel):
    mongodb: MongoDBConfig
    milvus: MilvusConfig
    embedding: EmbeddingConfig
    generator: GeneratorConfig

class ConfigLoader:
    # when a ConfigLoader instance is created, it reads the config.json file and converts it to a Pydantic model (Config)
    def __init__(self, config_path: str = "config.json"):
        #self.config_path = Path(config_path)
        self.config_path = Path(__file__).parent / "config.json"

        self.config: Config = self.load_config()

    def load_config(self) -> Config:
        # does the actual reading of the config file and conversion to Pydantic model
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as config_file:
            config_data = json.load(config_file)
        
        return Config(**config_data)

    def get_mongodb_config(self) -> MongoDBConfig:
        return self.config.mongodb

    def get_milvus_config(self) -> MilvusConfig:
        return self.config.milvus

    def get_embedding_config(self) -> EmbeddingConfig:
        return self.config.embedding

    def get_generator_config(self) -> GeneratorConfig:
        return self.config.generator