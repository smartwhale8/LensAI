from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from pymilvus import Collection, MilvusException
import logging

class EmbeddingHandler:
    def __init__(self, model_name: str, mongodb_handler, mivus_handler):
        self.model = SentenceTransformer(model_name)
        self.mongodb_handler = mongodb_handler
        self.milvus_handler = mivus_handler
        self.logger = logging.getLogger(__name__)

        #Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # get handle to mongodb collections, eg acts_collection, case_files_collection

    def generate_embeddings(self, texts):
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True) # Convert to tensor
            embeddings = embeddings.cpu().numpy() # Ensure embeddings are moved to CPU before converting to numpy array

        return embeddings
    
    def prepare_text_for_embedding(self, doc, collection_name):
        """Prepare text based on the collection name"""
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
            self.logger.error(f"Collection '{collection_name}' not found in Milvus, aborting...")
            return
        
        total_docs = self.mongodb_handler.count_documents(collection_name)
        print(f"Total documents in {collection_name} to process: {total_docs}")

        processed_docs = 0
        for i in range(0, total_docs, batch_size):
            # We cant really use projection here, as we want to be collection agnostic at this stage!
            docs = self.mongodb_handler.find(collection_name, {}, limit=batch_size, skip=i)
            batch_data = self.process_batch(docs, collection_name)
            if batch_data:
                try:
                    milvus_collection.insert(batch_data)
                    processed_docs += len(batch_data)
                    self.logger.info(f"Processed {processed_docs}/{total_docs} documents")
                except MilvusException as e:
                    self.logger.error(f"Failed to insert batch data into Milvus collection")
                    return False

        milvus_collection.flush()
        # index params define how the vectors are organized and stored in the DB.
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        milvus_collection.create_index("embedding", index_params)
        self.logger.info(f"Finished processing all documents and storing embeddings in Milvus for {collection_name}")
    
    def process_legal_acts_batch(self, acts: list) -> list:
        act_ids = []
        texts = []
        act_years = []
        short_titles = []
        long_titles = []

        for act in acts:
            act_ids.append(act["act_id"])
            texts.append(self.prepare_text_for_embedding(act, "legal_acts"))
            act_years.append(str(act.get("act_year", "")))
            short_titles.append(act.get("short_title", ""))
            long_titles.append(act.get("long_title", ""))
        
        embeddings = self.generate_embeddings(texts)
        return [act_ids, embeddings.tolist(), act_years, short_titles, long_titles]
    

    def process_case_files_batch(self, cases: list) -> list:
        case_ids = []
        texts = []
        case_dates = []
        case_types = []

        for case in cases:
            case_ids.append(case["act_id"])
            texts.append(self.prepare_text_for_embedding(case, "case_files"))
            case_dates.append(str(case.get("case_date", "")))
            case_types.append(case.get("case_type", ""))
        
        embeddings = self.generate_embeddings(texts)
        return [case_ids, embeddings.tolist(), case_dates, case_types]