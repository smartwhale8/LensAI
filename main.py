# for orchestrating the entire system

import argparse
import os
from tqdm import tqdm
import json

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

def count_pdf_files(folder_path):
    pdf_count = 0
    for root, dirs, files in os.walk(folder_path):
        print(f"root: {root}, and dirs: {dirs} with files: {len(files)}")
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_count += 1
    return pdf_count

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

    # Any remaining cleanup or processing
    print("Document ingestion completed.")


def main():
    parser = argparse.ArgumentParser(
        description='RAG System: A system for Retrieval-Augmented Generation.',
        epilog='Examples:\n'
               '  python main.py ingest case_files /path/to/folder\n'
               '  python main.py ingest legal_acts /path/to/folder\n'
               '\n'
               'For more information, use the help command for specific commands.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='commands')

    # Subparser for the ingestion command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into the database')
    ingest_parser.add_argument('file_type', type=str, choices=['case_files', 'legal_acts'], help='Type of files to ingest (case_files or legal_acts)')
    ingest_parser.add_argument('folder_path', type=str, help='Path to the folder containing documents to ingest')

    args = parser.parse_args()

    if args.command == 'ingest':
        if not args.folder_path or not args.file_type:
            ingest_parser.error("The 'folder_path' and 'file_type' arguments are required for the ingest command")
        run_document_ingestion(args.folder_path, args.file_type)
    elif args.command is None:
        # No command provided
        parser.print_help()
    else:
        #run_rag_pipeline()
        pass

if __name__ == "__main__":
    main()