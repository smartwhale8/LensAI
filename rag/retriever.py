import numpy as np
from typing import List, Dict
from rag.database.milvus_handler import MilvusHandler
from rag.database.mongodb_handler import MongoDBHandler
from rag.embedding.embedding_handler import EmbeddingHandler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize # ?? why this tokenizer was used!!!
from rag.embedding.rag_chunker import RAGChunker
from config.config import ConfigLoader


import torch
from utils.logger.logging_config import logger

#TODO: move it to config.py/config.json
class CollectionConfig:
    def __init__(self, name: str, id_field: str, chunk_id_field:str,
                 text_field: str, meta_fields: List[str], 
                 output_fields: List[str], chunk_size: int = 200, 
                 chunk_overlap: int = 50, vector_field: str = "embedding"):
        self.name = name
        self.id_field = id_field
        self.chunk_id_field = chunk_id_field
        self.text_field = text_field
        self.meta_fields = meta_fields
        self.output_fields = output_fields + [vector_field]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_field = vector_field

class CollectionConfigFactory:
    @staticmethod
    def get_config(collection_type: str) -> CollectionConfig:
        if collection_type == "legal_acts":
            return CollectionConfig(
                name="legal_acts",
                id_field="act_id",
                chunk_id_field="chunk_id",
                text_field="text_content", # Field containing the text content of the act in MongoDB
                meta_fields=["short_title", "long_title", "act_year"],
                output_fields=["short_title", "long_title", "act_year"],
                chunk_size=200,
                chunk_overlap=50
            )
        elif collection_type == "case_files":
            return CollectionConfig(
                name="case_files",
                id_field="case_id",
                text_field="text_content", # Field containing the text content of the case in MongoDB
                meta_fields=["court", "case_number", "year"],
                output_fields=["court", "case_number", "year"]
            )
        else:
            raise ValueError(f"Collection type '{collection_type}' not supported")

class Retriever:
    def __init__(
        self,
        milvus_handler: MilvusHandler,
        mongodb_handler: MongoDBHandler,
        embedding_handler: EmbeddingHandler,
        collection_config: CollectionConfig,
        top_k: int = 5,
        max_seq_length: int = ConfigLoader().get_embedding_config().max_seq_length,
    ):
        self.milvus = milvus_handler
        self.mongodb = mongodb_handler
        self.embedding = embedding_handler
        self.collection_config = collection_config
        self.top_k = top_k
        self.logger = logger
        if self.milvus.client is None:
            logger.error("Failed to connect to Milvus database")
            raise ConnectionError("Failed to connect to Milvus database")
        self.milvus.load_collection(collection_name=self.collection_config.name)
        self.query_text = None
        self.chunker = RAGChunker(
            chunk_size=max_seq_length,
            overlap=50
        )

    def retrieve(self, query_text: str, top_k: int = 4, threshold: float = 0.7) -> List[Dict[str, str]]:
        # Generate embedding for the query
        self.query_text = query_text # maintain the query text for later use
        query_embeddings = self.embedding.generate_embeddings([query_text])
        # Ensure that embeddings were generated
        if query_embeddings is None or len(query_embeddings) == 0:
            raise ValueError("Failed to generate query embeddings")

        # Extract the first query embedding
        query_embedding = query_embeddings[0]

        # Validate the shape of the embedding
        if query_embedding is None or query_embedding.size == 0:
            raise ValueError("Query embedding is empty")
        
        # Search similar documents in Milvus
        results = self.milvus.search(
            collection_name=self.collection_config.name,
            query_vectors=[query_embedding],
            top_k=self.top_k, # Fetch more results for filtering, maybe top_k * 2?
            field_name=self.collection_config.vector_field,
            search_params=self.milvus.get_default_search_params(collection_name=self.collection_config.name),
            output_fields=[self.collection_config.id_field] + self.collection_config.output_fields + [self.collection_config.vector_field]
        )

        # Fetch full document (act/case etc) details from MongoDB
        similar_docs = [] # list of dictionaries containing the similar documents
        for hits in results:
            for hit in hits:
                entity = hit.get('entity', {})
                try:
                    doc_id = entity.get(self.collection_config.id_field)
                except AttributeError:
                    logger.error(f"Warning: Could not find {self.collection_config.id_field} in hit")
                    continue

                # Initialize the similar doc we are preparing
                similar_doc = {
					"metadata": {
						self.collection_config.id_field: doc_id,
						"similarity_score": hit.get('distance')
					},
                    "text": None
                }
                
                # Safely add output fields to metadata
                for field in self.collection_config.output_fields:
                    try:
                        similar_doc["metadata"][field] = entity.get(field, f"Field '{field}' not found")
                    except AttributeError:
                        similar_doc["metadata"][field] = f"Field '{field}' not found"

                #print(f"Chunk is : {chunk}")
                
                # The full text is availabe in MongoDB, use the id field to fetch it
                # we will not send it to the generator, rather only use it for the 
                mongodb_doc = self.mongodb.find_one(
                    self.collection_config.name,
                    {self.collection_config.id_field: similar_doc["metadata"][self.collection_config.id_field]}
                )
                if mongodb_doc:
                    full_text = mongodb_doc.get(self.collection_config.text_field, "")
                else:
                    logger.warn(f"Warning: Document with id {doc_id} not found in MongoDB")

                # Get relevant excerpt using Milvus embeddings
                doc_embedding = np.array(entity.get(self.collection_config.vector_field, []))
                try:
                    similar_doc["relevant_excerpt"] = self._get_relevant_excerpt_using_rag_chunker(full_text, query_embedding, doc_embedding)
                    
                except ValueError as e:
                    logger.error(f"Error getting relevant excerpt for {doc_id}: {e}")
                    continue

                similar_docs.append(similar_doc)

        return similar_docs
    

    def _get_relevant_excerpt_using_rag_chunker(self, full_text, query_embedding, doc_embedding, max_tokens=512):
        if doc_embedding is None:
            raise ValueError("doc_embedding is None. Please ensure it's properly retrieved from Milvus.")
        
        if query_embedding is None:
            raise ValueError("query_embedding is None. Please ensure it's properly generated.")

        # Use RAGChunker to chunk the full text
        chunks = self.chunker.chunk_document({"text": full_text}, method='tokens')
        
        # Generate embeddings for all chunks
        chunk_embeddings = self.embedding.generate_embeddings([chunk['text'] for chunk in chunks])

        # Calculate cosine similarities with the document embedding
        doc_similarities = np.dot(chunk_embeddings, doc_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(doc_embedding)
        )
        
        # Calculate cosine similarities with the query embedding
        query_similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Combine both similarities
        combined_similarities = 0.5 * doc_similarities + 0.5 * query_similarities
        
        # Sort chunks by similarity
        sorted_indices = combined_similarities.argsort()[::-1]
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ConfigLoader().get_generator_config().gen_tokenizer)

        # Count tokens in the query
        query_tokens = len(tokenizer.tokenize(self.query_text))
        relevant_chunks = []
        total_tokens = query_tokens

        # Select chunks while respecting token limit
        for idx in sorted_indices:
            chunk_tokens = len(tokenizer.tokenize(chunks[idx]['text']))
            if total_tokens + chunk_tokens > max_tokens:
                break
            relevant_chunks.append(chunks[idx]['text'])
            total_tokens += chunk_tokens                
        
        # Join the selected chunks
        relevant_excerpt = ' '.join(relevant_chunks)

        return relevant_excerpt