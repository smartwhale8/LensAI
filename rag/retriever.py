import numpy as np
from typing import List, Dict
from rag.database.milvus_handler import MilvusHandler
from rag.database.mongodb_handler import MongoDBHandler
from rag.embedding.embedding_handler import EmbeddingHandler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import torch
import logging

class CollectionConfig:
    def __init__(self, name: str, id_field: str, text_field: str, meta_fields: List[str], output_fields: List[str], chunk_size: int = 200, chunk_overlap: int = 50, vector_field: str = "embedding"):
        self.name = name
        self.id_field = id_field
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
        top_k: int = 5
    ):
        self.milvus = milvus_handler
        self.mongodb = mongodb_handler
        self.embedding = embedding_handler
        self.collection_config = collection_config
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)
        if self.milvus.client is None:
            raise ConnectionError("Failed to connect to Milvus database")
        self.milvus.load_collection(collection_name=self.collection_config.name)
        self.query_text = None

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
        
        # Search params determine how the search is performed; influence the accuracy and efficiency of the search; Search metric type must match the metric used during indexing, 'IP' for our case
        #TODO #[CodeReview]: Move the search config to a config file, it's too critical to be hardcoded and hidden here
        search_params = {
            "metric_type": "IP", 
            "params": {"nprobe": 10}
        }

        # Search similar documents in Milvus
        results = self.milvus.search(
            collection_name=self.collection_config.name,
            query_vectors=[query_embedding],
            top_k=self.top_k, # Fetch more results for filtering, maybe top_k * 2?
            field_name=self.collection_config.vector_field,
            search_params=search_params,
            output_fields=[self.collection_config.id_field] + self.collection_config.output_fields + [self.collection_config.vector_field]
        )

        # Fetch full document (act/case etc) details from MongoDB
        similar_docs = [] # list of dictionaries containing the similar documents
        for hits in results:
            for hit in hits:
                entity = hit.get('entity', {})
                try:
                    act_id = entity.get(self.collection_config.id_field)
                except AttributeError:
                    logging.error(f"Warning: Could not find {self.collection_config.id_field} in hit")
                    continue

                # Initialize the similar doc we are preparing
                similar_doc = {
					"metadata": {
						self.collection_config.id_field: act_id,
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
                    logging.error(f"Warning: Document with id {act_id} not found in MongoDB")

                # Get relevant excerpt using Milvus embeddings
                doc_embedding = np.array(entity.get(self.collection_config.vector_field, []))
                try:
                    similar_doc["relevant_excerpt"] = self._get_relevant_excerpt_using_milvus_embedding_with_chunking(full_text, query_embedding, doc_embedding)
                    #self._get_relevant_excerpt_using_milvus_embedding(full_text, query_embedding, doc_embedding)
                    
                except ValueError as e:
                    logging.error(f"Error getting relevant excerpt for {act_id}: {e}")
                    continue

                similar_docs.append(similar_doc)

        return similar_docs
    

    def chunk_sentences(self, sentences, chunk_size=3):
        return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    def _get_relevant_excerpt_using_milvus_embedding_with_chunking(self, full_text, query_embedding, doc_embedding, num_chunks=2, chunk_size=2, max_tokens=512):
        """
        1. Use sentence tokenization to break full text into sentences.
        2. Chunk these sentences into groups of a specified size (default: 3 sentences per chunk).
        3. Generate embedding for each chunk.
        4. Calculate cosine similarity between chunks and both the document embeddings and query embeddings.
        5. Select the top most relevant chunks based on the combined similarity scores.
        6. Join the top chunks to form the relevant excerpt.
        """
        if doc_embedding is None:
            raise ValueError("doc_embedding is None. Please ensure it's properly retrieved from Milvus.")
        
        if query_embedding is None:
            raise ValueError("query_embedding is None. Please ensure it's properly generated.")

        sentences = sent_tokenize(full_text)
        chunks = self.chunk_sentences(sentences, chunk_size)
        
        # Generate embeddings for all chunks using GPU if available
        chunk_embeddings = self.embedding.generate_embeddings(chunks)

        # Ensure query_embedding is a numpy array
        query_embedding = np.array(query_embedding)
        
        # Calculate cosine similarities with the document embedding
        doc_similarities = np.dot(chunk_embeddings, doc_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(doc_embedding)
        )
        
        # Calculate cosine similarities with the query embedding
        query_similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Combine both similarities (you can adjust the weights)
        combined_similarities = 0.5 * doc_similarities + 0.5 * query_similarities
        
        # Sort chunks by similarity
        sorted_indices = combined_similarities.argsort()[::-1]
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        # Count tokens in the query
        query_tokens = len(tokenizer.tokenize(self.query_text))
        relevant_chunks = []
        total_tokens = query_tokens

        # Select chunks while respecting token limit
        for idx in sorted_indices:
            chunk_tokens = len(tokenizer.tokenize(chunks[idx]))
            if total_tokens + chunk_tokens > max_tokens:
                break
            relevant_chunks.append(chunks[idx])
            total_tokens += chunk_tokens                
        
        # Join the selected chunks
        relevant_excerpt = ' '.join(relevant_chunks)

        return relevant_excerpt


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
        sentence_embeddings = self.embedding.generate_embeddings(sentences)
        
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
