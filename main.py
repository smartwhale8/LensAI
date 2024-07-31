# for orchestrating the entire system

import argparse
import os
from tqdm import tqdm
import json
from logging_config import setup_logging
import logging

class RAGOrchestrator:
    def __init__(self, mongodb, milvus, embedding, retriever, generator):
        self.mongodb = mongodb
        self.milvus = milvus
        self.embedding = embedding
        self.retriever = retriever
        self.generator = generator

    def setup(self):
        # Set up MongoDB and Milvus, create collections and indexes
        # MongoDb: legal acts collection: "legal_acts"
        self.mongodb.create_collection("legal_acts")

        # Milvus: legal acts collection: "legal_acts" <- same name as MongoDB collection, coz why not!
        self.milvus.create_collection("legal_acts")
        pass

    def index_documents(self):
        # Create embeddings and index documents in Milvus
        pass

    def query(self, user_query):
        # Process user query through RAG Pipeline.
        # Retrieve relevant documents based on user query
        pass

    def interaction_loop(self):
        # Main interaction loop for the RAG system
        pass

    ##TODO: maybe we don't need this, as we want more granular control over the individual processes.
    def run(self):
        self.mongodb.connect()
        self.milvus.connect()
        self.generator.generate_data()
        self.embedding.generate_embeddings()
        self.retriever.retrieve()

    def shutdown(self):
        self.mongodb.close()
        self.milvus.close()

def run_rag_pipeline():
    # Lazy imports
    from rag.database.mongodb_handler import MongoDBHandler
    from rag.database.milvus_handler import MilvusHandler
    from rag.embedding.embedding_handler import EmbeddingHandler
    from rag.retriever import Retriever, CollectionConfig, CollectionConfigFactory
    from rag.generator import Generator

    # Load configuration
    #config = load_config()

    # Initialize Handlers
    mongodb = MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai")
    milvus = MilvusHandler(host="localhost", port="19530") 
    embedding = EmbeddingHandler("sentence-transformers/all-mpnet-base-v2", mongodb, milvus)
    
    #Initialize RAG components
    retriever = Retriever(
        milvus, 
        mongodb, 
        embedding, 
        CollectionConfigFactory.get_config("legal_acts")
    )
    generator = Generator()

    # Run the retreival and generation loop

def run_document_ingestion(folder_path, file_type):
    from rag.database.mongodb_handler import MongoDBHandler  # Lazy import
    from ingestion.document_ingestion import DocumentIngestion  # Lazy import

    # Check if the folder_path exists and is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' is not valid.")
        return

    with MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai") as mongodb_handler:
        # Initialize the ingestion handler
        ingestion_handler = DocumentIngestion(mongodb_handler, folder_path, file_type)
        ingestion_handler.ingest_from_folder()
        #close() called automatically on mongodb_handler after exiting this context

    # Any remaining cleanup or processing
    print("Document ingestion completed.")

def run_embedding_generation():
    from rag.database.mongodb_handler import MongoDBHandler  # Lazy import
    from rag.database.milvus_handler import MilvusHandler  # Lazy import
    from rag.embedding.embedding_handler import EmbeddingHandler  # Lazy import
    logger = logging.getLogger(__name__)

    with MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai") as mongodb_handler, \
         MilvusHandler(host='localhost', port='19530') as milvus_handler:
        
        # Create mivlus collection
        milvus_handler.create_collection("legal_acts", force=True)
        
        # Initialize the embedding generator
        embedding_generator = EmbeddingHandler(
             "sentence-transformers/all-mpnet-base-v2", 
             mongodb_handler, 
             milvus_handler
             )
        # Generate and store embeddings for all documents in the legal_acts collection of MongoDB
        embedding_generator.process_and_store_embeddings("legal_acts")


def main():
    setup_logging(log_level='INFO')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='RAG System: A system for Retrieval-Augmented Generation.',
        epilog='Examples:\n'
               '  python main.py ingest case_files /path/to/folder\n'
               '  python main.py ingest legal_acts /path/to/folder\n'
               '  python main.py generate_embeddings\n'
               '\n'
               'For more information, use the help command for specific commands.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='commands')

    # Subparser for the ingestion command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into the database')
    ingest_parser.add_argument('file_type', type=str, choices=['case_files', 'legal_acts'], help='Type of files to ingest (case_files or legal_acts)')
    ingest_parser.add_argument('folder_path', type=str, help='Path to the folder containing documents to ingest')

    # Subparser for the generate_embeddings command
    generate_embeddings_parser = subparsers.add_parser('generate_embeddings', help='Generate embeddings for the ingested documents')

    # Common argument for log level
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()

    try:
        if args.command == 'ingest':
            if not args.folder_path or not args.file_type:
                ingest_parser.error("The 'folder_path' and 'file_type' arguments are required for the ingest command")
            run_document_ingestion(args.folder_path, args.file_type)
        elif args.command == 'generate_embeddings':
            run_embedding_generation()
        elif args.command is None:
            # No command provided
            parser.print_help()
        # else:
        #     #run_rag_pipeline()
        #     pass
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("An error occurred: %s", str(e), exc_info=True)
        print(f"An error occurred: {e}. Please check the log file for more details.")

if __name__ == "__main__":
    main()