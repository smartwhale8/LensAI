# for orchestrating the entire system

import argparse
import os
from tqdm import tqdm
import json
from typing import List, Tuple
#from utils.logging_config import setup_logging
#import logging
from utils.logger.logging_config import logger
from colorama import Fore, Style, Back, init


## tODO: move this to a better place:
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from config.config import ConfigLoader
from rag.embedding.rag_chunker import RAGChunker


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

def get_context(retreived_docs, max_tokens=2048):
    context = ""
    for doc in retreived_docs:
        excerpt = doc["relevant_excerpt"]
        if len(context) + len(excerpt) > max_tokens:
            break
        context += f"\n\nSource {doc['metadata']['act_id']}:\n{excerpt}"
    return context.strip()

def run_rag_pipeline():
    # Lazy imports
    from rag.database.mongodb_handler import MongoDBHandler
    from rag.database.milvus_handler import MilvusHandler
    from rag.embedding.embedding_handler import EmbeddingHandler
    from rag.retriever import Retriever, CollectionConfig, CollectionConfigFactory
    from rag.generator import Generator
    from rag.rag_response import RAGResponse

    # Load configuration
    #config = load_config()

    with MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai") as mongodb_handler, MilvusHandler() as milvus_handler:
        #Initialize RAG components        

        """
        Create embedding model : sentence-transformers/all-mpnet-base-v2
        Create Chunker object, passing it the embedding model
        Create Retriever object, passing it the Milvus and MongoDB handlers, the embedding handler, and the collection config
        Create Generator object
        """
        #TODO: Create a class that creates Models
        # Create the Embedding model
        config = ConfigLoader().get_embedding_config()
        transformer = Transformer(config.emb_model_name, config.max_seq_length)
        pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
        emb_model = SentenceTransformer(modules=[transformer, pooling]) # The Embedding model

        # Create the Chunker object
        chunker = RAGChunker(emb_model=emb_model, chunk_size=config.max_seq_length, overlap=50, store_tokens=False) #store_toekns is True only when we want to store embeddings to Mivlus collection, not for 'inference'
        
        embedding_handler = EmbeddingHandler(emb_model, chunker, mongodb_handler, milvus_handler)

        retriever = Retriever(
            milvus_handler, 
            mongodb_handler, 
            embedding_handler,
            chunker,
            CollectionConfigFactory.get_config("legal_acts")
        )
        generator = Generator()

        # Run the retreival and generation loop

        chat_history: List[Tuple[str, str]] = []
        print(Fore.BLUE + Back.WHITE + "Welcome to the Legal Research Assistant 'LegalGenie'!\n Type 'exit', 'bye' or 'end' to finish the conversation.")
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() in ['exit', 'bye', 'end']:
                print(Fore.GREEN + "Thank you for using the Legal Research Assistant. Goodbye!")
                break

            # Retrieve relevant documents (acts/case files) based on user query
            retrieved_docs = retriever.retrieve(query_text=user_query)
            if not retrieved_docs:
                print(Fore.RED + "LegalGenie: I'm sorry, I couldn't find any relevant documents based on your query.")
                continue
            else:
                print(Fore.YELLOW + f"Retrieved {len(retrieved_docs)} documents.")
                context = get_context(retrieved_docs)
                
            # Generate response
            #context = "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history[-3:]])  # Use last 3 exchanges as context ?
            response = generator.generate(user_query, context)

            # Create the RAGResponse object
            rag_response = RAGResponse(response, retrieved_docs)
            # Use the RAGResponse methods as needed
            print("==============================")
            print(rag_response.get_formatted_response())
            metadata = rag_response.get_metadata()
            json_response = rag_response.get_json_response()
            
            print("metadata: ", metadata) # Metadata for the response
            print("json_response: ", json_response) # JSON response
            
            print("==============================")
            chat_history.append(("User", user_query))
            chat_history.append(("LegalGenie", response))
            print(Fore.WHITE + Back.GREEN + f"LegalGenie: {response}")


def run_document_ingestion(folder_path, file_type):
    from rag.database.mongodb_handler import MongoDBHandler  # Lazy import
    from ingestion.document_ingestion import DocumentIngestion  # Lazy import

    # Check if the folder_path exists and is a directory
    if not os.path.isdir(folder_path):
        logger.error(f"Error: The folder path '{folder_path}' is not valid.")
        return

    with MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai_hybrid") as mongodb_handler:
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

    with MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai_hybrid") as mongodb_handler, \
         MilvusHandler() as milvus_handler:
        
        # Create mivlus collection
        milvus_handler.create_collection("legal_acts", force=True)
        # Create the Embedding model
        config = ConfigLoader().get_embedding_config()
        transformer = Transformer(config.emb_model_name, config.max_seq_length)
        pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
        emb_model = SentenceTransformer(modules=[transformer, pooling]) # The Embedding model

        # Create the Chunker object
        chunker = RAGChunker(emb_model=emb_model, chunk_size=config.max_seq_length, overlap=50, store_tokens=False) #store_toekns is True only when we want to store embeddings to Mivlus collection, not for 'inference'
        
        # Initialize the embedding generator
        embedding_generator = EmbeddingHandler(mongodb_handler, milvus_handler, emb_model, chunker)
        # Generate and store embeddings for all documents in the legal_acts collection of MongoDB
        logger.info("Generating embeddings for legal acts...")
        embedding_generator.process_and_store_embeddings("legal_acts", batch_size=32)


def main():

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

    # Subparser for the process_query command
    process_query_parser = subparsers.add_parser('process_query', help='Process user query through RAG Pipeline')
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
        elif args.command == 'process_query':
            run_rag_pipeline()
        elif args.command is None:
            # No command provided
            parser.print_help()
        # else:
        #     #run_rag_pipeline()
        #     pass
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)

if __name__ == "__main__":
    main()