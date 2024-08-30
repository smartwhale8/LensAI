from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from config.config import ConfigLoader
from .rag_chunker import RAGChunker
from pymilvus import MilvusException, DataType, FieldSchema, Collection
from utils.logger.logging_config import logger
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Dict
import numpy as np
import logging
import torch

class EmbeddingHandler:
    def __init__(self,
                 mongodb_handler, 
                 mivus_handler,
                 emb_model: SentenceTransformer,
                 chunker : RAGChunker
        ):
        #self.logger = logger
        #self.config = ConfigLoader().get_embedding_config()
        #transformer = Transformer(self.config.emb_model_name, self.config.max_seq_length)
        #pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
        #self.model = SentenceTransformer(modules=[transformer, pooling]) # The Embedding model
        self.model = emb_model
        self.mongodb_handler = mongodb_handler
        self.milvus_handler = mivus_handler
        self.chunker = chunker
        #Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Initialize the RAGChunker
        
        # get handle to mongodb collections, eg acts_collection, case_files_collection

    def generate_embeddings(self, texts):
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_tensor=True) # Convert to tensor
            embeddings = embeddings.cpu().numpy() # Ensure embeddings are moved to CPU before converting to numpy array

        return embeddings
    
    def validate_batch_data(self, batch_data: List[List], schema: List[FieldSchema]) -> List[List]:
        """
        Validate the batch data against the schema
        """

        if not batch_data or len(batch_data) != len(schema):
            logger.error(f"Invalid batch_data format. Expected {len(schema)} fields, got {len(batch_data)}")
            return None

        num_items = len(batch_data[0])
        validated_batch = []

        for i in range(num_items):
            valid_item = []
            for field_data, field_schema in zip(batch_data, schema):
                if i >= len(field_data):
                    logger.error(f"Insufficient data for field '{field_schema.name}' at index {i}")
                    return None

                value = field_data[i]

                if field_schema.dtype == DataType.INT64:  # act_id and chunk_id
                    if not isinstance(value, int):
                        logger.error(f"Field '{field_schema.name}' at index {i} must be of type int, got {type(value)}")
                        return None
                elif field_schema.dtype == DataType.FLOAT_VECTOR:  # embedding
                    if not isinstance(value, list) or len(value) != field_schema.dim:
                        logger.error(f"Field '{field_schema.name}' at index {i} must be a list of {field_schema.dim} floats")
                        return None
                    if not all(isinstance(v, (int, float)) for v in value):
                        logger.error(f"Field '{field_schema.name}' at index {i} contains non-numeric values")
                        return None
                elif field_schema.dtype == DataType.VARCHAR:  # act_year, short_title, long_title
                    if not isinstance(value, str):
                        logger.error(f"Field '{field_schema.name}' at index {i} must be a string, got {type(value)}")
                        return None
                    if len(value) > field_schema.max_length:
                        logger.error(f"Field '{field_schema.name}' at index {i} exceeds max length of {field_schema.max_length}")
                        return None
                
                valid_item.append(value)

            validated_batch.append(valid_item)

        return validated_batch

    def insert_batch_data_into_milvus(self, collection: Collection, batch_data: List[List], schema: List[FieldSchema]) -> bool:
        if len(batch_data) != len(schema):
            logger.error(f"Mismatch between batch_data fields ({len(batch_data)}) and schema fields ({len(schema)})")
            return False

        num_items = len(batch_data[0])
        # Prepare the data in the format Milvus expects (as columns!)
        insert_data = []
        for i, field_schema in enumerate(schema):
            field_data = batch_data[i]
            
            # Handle FLOAT_VECTOR separately
            if field_schema.dtype == DataType.FLOAT_VECTOR:
                insert_data.append(field_data)
            else:
                insert_data.append(field_data)

        #TODO: Validate batches?
        # validated_batch = self.validate_batch_data(batch_data, schema)
        # if validated_batch is None:
        #     logger.error("Batch validation failed. No data inserted.")
        #     return False

        try:
            result = collection.insert(insert_data)
            logger.info(f"Successfully inserted batch of {num_items} items into Milvus")
            logger.debug(f"Insert result: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch data into Milvus: {e}")
            logger.error(f"Error type: {type(e)}")
            return False
        
    def prepare_text_for_embedding(self, doc, collection_name):
        """
        Prepare text based on the collection name
        This method combines various fields of a document into a single text string. TODO: check if we need to change/improve this strategy
        This string is then passed to the embedding model to generate embeddings.
        """
        #logger.info(f"Preparing text for embedding for document {doc.get('act_id', 'N/A')} in collection '{collection_name}'")

        if collection_name == "legal_acts":
            text_parts = [
                f"ID: {doc.get('act_id', 'N/A')}",
                f"Content: {doc.get('text_content', 'N/A')}",        
                f"Year: {doc.get('act_year', 'N/A')}",
                f"Short Title: {doc.get('short_title', 'N/A')}",
                f"Long Title: {doc.get('long_title', 'N/A')}"
            ]
            return ' | '.join(filter(lambda x: 'N/A' not in x, text_parts))
        elif collection_name == "case_files":
            text_parts = [
                f"Case ID: {doc.get('case_id', 'N/A')}",
                f"Description: {doc.get('case_description', 'N/A')}",        
                f"Date: {doc.get('case_date', 'N/A')}",
                f"Type: {doc.get('case_type', 'N/A')}"
            ]
            return ' | '.join(filter(lambda x: 'N/A' not in x, text_parts))
        # Add more collections here as needed as elif blocks
        else:
            raise ValueError(f"Collection '{collection_name}' not recognized")

    def process_batch(self, docs: list, collection_name: str) -> list:
        if collection_name == "legal_acts":
            return self.process_legal_acts_batch(docs)
        elif collection_name == "case_files":
            return self.process_case_files_batch(docs)
        else:
            raise ValueError(f"Collection '{collection_name}' not recognized")

    def process_and_store_embeddings(self, collection_name: str, batch_size: int = 32):
        milvus_collection = self.milvus_handler.get_collection(collection_name)

        if not milvus_collection:
            logger.error(f"Collection '{collection_name}' not found in Milvus, aborting...")
            return
        
        total_docs = self.mongodb_handler.count_documents(collection_name)

        processed_docs = 0
        for i in range(0, total_docs, batch_size):
            # We cant really use projection here, as we want to be collection agnostic at this stage!
            if collection_name == "legal_acts":
                projection = {"act_id": 1, "text_content": 1, "act_year": 1, "short_title": 1, "long_title": 1}
            elif collection_name == "case_files":
                projection = {"case_id": 1, "case_description": 1, "case_date": 1, "case_type": 1}

            docs = self.mongodb_handler.find_documents(collection_name, filter_dict=None, projection=projection, skipi=i, limit=batch_size)
            # print len of docs
            #logger.info(f"Processing batch {i+1}-{i+batch_size} of {total_docs} documents")
            batch_data = self.process_batch(docs, collection_name)
            if batch_data:
                success = self.insert_batch_data_into_milvus(milvus_collection, batch_data, schema=self.milvus_handler.get_schema(collection_name))
                if success:
                    processed_docs += len(docs)
                    logger.info(f"Processed {processed_docs}/{total_docs} documents")
                else:
                    logger.error(f"Failed to process batch {i+1}-{i+batch_size}")
                    return False
        
        milvus_collection.flush()
        milvus_collection.create_index("embedding", self.milvus_handler.get_index_params(collection_name))
        logger.info(f"Finished processing all documents ({processed_docs}) and storing embeddings in Milvus for {collection_name}")
    
    def process_legal_acts_batch(self, acts: list) -> list:
        logger.info(f"Processing batch of {len(acts)} legal acts...")
        act_ids = []
        chunk_ids = []
        texts = []
        act_years = []
        short_titles = []
        long_titles = []
        
        with tqdm(total=len(acts), desc="Overall progress", unit="act", leave=False) as pbar:
            for act in acts:
                original_text = self.prepare_text_for_embedding(act, "legal_acts")
                chunks = self.chunker.chunk_document({"text": original_text}, method='hybrid')
                for chunk in chunks:
                    act_ids.append(act['act_id'])
                    chunk_ids.append(chunk['chunk_id'])
                    texts.append(chunk["text"])
                    act_years.append(str(act.get("act_year", "")))
                    short_titles.append(act.get("short_title", ""))
                    long_titles.append(act.get("long_title", ""))
                pbar.update(1)
                    
        embeddings = self.generate_embeddings(texts)
    
        return [act_ids, chunk_ids, embeddings.tolist(), act_years, short_titles, long_titles]
    

    def process_case_files_batch(self, cases: list) -> list:
        case_ids = []
        texts = []
        case_dates = []
        case_types = []

        for case in cases:
            original_text = self.prepare_text_for_embedding(case, "case_files")
            chunks = self.chunker.chunk_document({"text": original_text}, method='tokens')

            for chunk in chunks:
                case_ids.append(f"{case['case_id']}_{chunk['chunk_id']}")
                texts.append(chunk["text"])
                case_dates.append(str(case.get("case_date", "")))
                case_types.append(case.get("case_type", ""))
        
        embeddings = self.generate_embeddings(texts)  # Generate embeddings for the text chunks
        return [case_ids, embeddings.tolist(), case_dates, case_types]