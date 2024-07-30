import os
import json
from tqdm import tqdm
from PyPDF2 import PdfReader
from typing import Any, List, Dict
from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError
from rag.database.mongodb_handler import MongoDBHandler

class DocumentIngestion:
    """
    Data Ingestion Module:
        APIs: Expose APIs for inserting documents.
        Validation: Validate incoming data.
        Preprocessing: Preprocess and transform data as needed.
        Storage: Insert validated and processed documents into MongoDB.
    """
    def __init__(self, mongodb_handler: MongoDBHandler, folder_path: str, file_type: str):
        self.mongodb_handler = mongodb_handler
        self.folder_path = folder_path
        self.file_type = file_type

    def ingest_from_folder(self):
        if self.file_type == "legal_acts":
            for year_folder in os.listdir(self.folder_path):
                year_path = os.path.join(self.folder_path, year_folder)
                if os.path.isdir(year_path):
                    print(f"Processing year: {year_folder}")
                    self.process_yearly_acts(year_path)
        elif self.file_type == "case_files":
            # TODO: Implement case files ingestion
            pass

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def is_empty(self, value):
        if value is None or value == "":
            return True
        if isinstance(value, (list, dict)) and not value:
            return True
        return False
    
    def extract_metadata_from_text(self, text):
        def is_valid_date(date_string):
            try:
                datetime.strptime(date_string, '%Y-%m-%d')
                return True
            except ValueError:
                return False

        lines = text.strip().split('\n')
        metadata = {}
        for line in lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip()
                
                if key in ['enactment_date', 'enforcement_date']:
                    if is_valid_date(value):
                        value = datetime.strptime(value, '%Y-%m-%d')
                elif key in ['act_id', 'act_number', 'act_year']:
                    numeric_value = ''.join(filter(str.isdigit, value))
                    if numeric_value:
                        value = int(numeric_value)
                
                metadata[key] = value
        return metadata

    def insert_document(self, collection_name: str, metadata: Dict[str, Any], pdf_filename: str, text_content: str):
        if collection_name == "legal_acts":
            act_data = {
                "act_id": metadata.get("act_id", ""),
                "act_number": metadata.get("act_number", ""),
                "enactment_date": metadata.get("enactment_date", None),
                "act_year": metadata.get("act_year", ""),
                "short_title": metadata.get("short_title", ""),
                "long_title": metadata.get("long_title", ""),
                "ministry": metadata.get("ministry", ""),
                "department": metadata.get("department", ""),
                "enforcement_date": metadata.get("enforcement_date", None),
                "pdf_file_path": pdf_filename,
                "text_content": text_content,
                "last_updated": datetime.now(timezone.utc)
            }

            # Remove keys with None values or empty strings
            act_data = {k: v for k, v in act_data.items() if not self.is_empty(v)}

            try:
                result = self.mongodb_handler.upsert_document(
                    collection_name, 
                    {"act_id": metadata["act_id"]},
                    act_data,
                )
                if result.matched_count > 0:
                    return "updated", result.modified_count
                elif result.upserted_id:
                    return "inserted", result.upserted_id
                else:
                    return "no_change", 0
            except DuplicateKeyError as e:
                # This shouldn't happen with update_one and upsert, but just in case
                print(f"Duplicate act_id found for {metadata['act_id']}. Updating existing document.")
                result = self.mongodb_handler.update_one(
                    collection_name,
                    {"act_id": metadata["act_id"]},
                    {"$set": act_data}
                )
                return "updated", result.modified_count

        elif collection_name == "case_files":
            # TODO: Implement case files insertion
            pass

    def process_yearly_acts(self, year_folder: str):
        metadata_path = os.path.join(year_folder, "metadata.json")
        acts_folder = os.path.join(year_folder, "act_pdfs")

        with open(metadata_path, 'r', encoding='utf-8') as file:
            metadata_list = json.load(file)

        for metadata in tqdm(metadata_list):
            pdf_filename = metadata['pdf_filename']
            pdf_path = os.path.join(acts_folder, pdf_filename)
            text_content = self.extract_text_from_pdf(pdf_path)

            metadata = self.extract_metadata_from_text(metadata['metadata_text'])
            act_id = str(metadata.get('act_id', ''))

            if not act_id:
                print(f"Skipping act with no act_id: {metadata.get('short_title', 'Unknown Title')}")
                continue

            result = self.insert_document("legal_acts", metadata, pdf_filename, text_content)
            if result:
                print(f"Inserted/Updated act {pdf_filename} in MongoDB")
            else:
                print(f"Failed to insert/update act {pdf_filename}")

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract and process documents from a file.
        Parameters:
        - file_path: Path to the file containing the documents.
        Returns:
        - A list of dictionaries, where each dictionary represents a document.
        """
        pass

    def ingest(self, collection_name: str, documents: List[Dict], id_field: str):
        """
        Insert of update multiple documents into specified MongoDB collection.

        Parameters:
        - collection_name: Name of the MongoDB collection.
        - documents: List of document dictionaries to be inserted or updated.
        - id_field: The field used to identify documents uniquely (used for matching).
        """
        collection = self.mongodb_handler.db[collection_name]

        # Prepare bulk write operations
        operations = []
        for doc in documents:
            filter_dict = {id_field: doc[id_field]}
            operations.append({
                'update_one': {
                    'filter': filter_dict,
                    'update': {'$set': doc},
                    'upsert': True
                }
            })
        self.mongodb.bulk_upsert("legal_acts", documents)
        pass

    def update(self, document: Dict[str, Any]) -> None:
        """
        Update a single document in the collection, typically used for:
            1. fields of an existing document, e.g. updating the content of a legal act or a case file.
            2. Partial updates: updating only a few fields of a document without affecting the rest, when new details of the case are available.
        """
        pass

    def delete(self, document_id: str) -> None:
        """
        Delete a single document from the collection, typically used for:
            1. Removing a legal act (repealed?) or a outdated case file from the database, because we do not want to consider them in the search results.
            2. Sometimes a document may be deleted due to data quality or privacy concerns, or maybe due to a user request.
        """
        pass

    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the collection to retrieve documents based on the query.
        Parameters:
        - query: A dictionary containing the query parameters.

        Returns:
        - A list of documents that match the query.
        """
        pass