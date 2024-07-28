import numpy as np
from typing import List, Dict
from rag.database.milvus_handler import MilvusHandler
from rag.database.mongodb_handler import MongoDBHandler
from rag.embedding.embedding_handler import EmbeddingHandler
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import torch

class CollectionConfig:
    name: str
    id_field: str
    text_field: str
    meta_fields: List[str]
    output_fields: List[str]
    chunk_size: int = 200
    chunk_overlap: int = 50
    vector_field: str = "embedding"

class Retriever:
    def __init__(
        self,
        milvus_handler: MilvusHandler,
        mongodb_handler: MongoDBHandler,
        embedding_handler: EmbeddingHandler,
        collection_config: CollectionConfig,
        top_k: int = 5
    ):
        self.milvus = milvus_handler
        self.mongodb = mongodb_handler
        self.embedding = embedding_handler
        self.config = collection_config
        self.top_k = top_k

    
    ##TODO: Implement similarity search using the LG_RAG_Demo.ipynb notebook and return similar acts (as well as case files)
    
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        # Embed the query
        query_embedding = self.embedding.generate_embeddings(query)
        
        # Search params determine how the search is performed; influence the accuracy and efficiency of the search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Search similar documents in Milvus
        results = self.milvus.search(
            collection_name=self.config.name,
            vectors=[query_embedding],
            top_k=self.top_k,
            field_name=self.config.vector_field,
            search_params=search_params,
            output_fields=self.config.id_field + self.config.output_fields
        )

        # Fetch full document details from MongoDB
        retrieved_chunks = []
        for hits in results:
            for hit in hits:
                chunk = {
                    "text": hit.entity.get(self.config.text_field, ""),
                    "metadata": {
                        self.config.id_field: hit.entity.get(self.config.id_field),
                        "similarity_score": hit.distance
                    }
                }
                for field in self.config.output_fields:
                    chunk["metadata"][field] = hit.entity[field]

                # If text field is not in Milvus, fetch from MongoDB
                if not chunk["text"]:
                    mongodb_doc = self.mongodb.find_one(
                        self.config.name,
                        {self.config.id_field: chunk["metadata"][self.config.id_field]}
                    )
                    if mongodb_doc:
                        chunk["text"] = mongodb_doc.get(self.config.text_field, "")

                retrieved_chunks.append(chunk)
    
        return retrieved_chunks
    

def chunk_sentences(sentences, chunk_size=3):
    return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def _get_relevant_excerpt_using_milvus_embedding(self, full_text, query_embedding, doc_embedding, context_sentences=3):
    """
    Key idea: 
    To find sentences that are most similar to both the overall document topic (represented by the document embedding)
      and the specific query, providing a relevant excerpt that balances document context and query relevance.
    
    Calculate two sets of cosine similarities:
        a. Between each sentence embedding and the document embedding.
        b. Between each sentence embedding and the query embedding.
    Combine these similarities with equal weights (0.5 each) to get a final similarity score for each sentence.
    
    context_sentences: the function will return an excerpt consisting of the three most relevant sentences (Default: 3)
    """
    sentences = sent_tokenize(full_text)

    # Generate embeddings for all sentences using GPU if available
    sentence_embeddings = self.embedding.generate_embedding(sentences)
    
    # Ensure query_embedding is a numpy arrays; doc_embedding is already one
    query_embedding = np.array(query_embedding)
    
    # Calculate cosine similarities with the document embedding
    doc_similarities = np.dot(sentence_embeddings, doc_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(doc_embedding)
    )

    # Calculate cosine similarities with the query embedding
    query_similarities = np.dot(sentence_embeddings, query_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Combine both similarities (one can adjust the weights)
    combined_similarities = 0.5 * doc_similarities + 0.5 * query_similarities

    # Get indices of top similar sentences
    top_indices = combined_similarities.argsort()[-context_sentences:][::-1]

    # Sort indices to maintain original order
    top_indices = sorted(top_indices)

    # Join the top sentences
    relevant_excerpt = ' '.join([sentences[i] for i in top_indices])

    return relevant_excerpt
