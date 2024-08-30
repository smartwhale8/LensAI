from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
from config.config import ConfigLoader
from typing import List, Dict, Union
from utils.logger.logging_config import logger
import torch
import re

class RAGChunker:
    """
    Here, two different tokenizers are used for different purposes, respsctively:
        1. The embedding tokenizer is used to process and chunk text into manageable pieces for generating embeddings, ensuring that the chunks do not exceed the maximum sequence length of the embedding model.
        2. The generation tokenizer is used to prepare the context and query inputs for text generation, ensuring that the combined token length of the context and queries remains within the generator model's maximum context window.
    """
    def __init__(self,
            emb_model = SentenceTransformer, 
            gen_tok_name = ConfigLoader().get_generator_config().gen_tokenizer,
            chunk_size: int = ConfigLoader().get_embedding_config().max_seq_length, 
            overlap: int = 50,
            store_tokens: bool = False):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the embedding model and tokenizer
        self.emb_model = emb_model
        self.emb_tok = AutoTokenizer.from_pretrained(ConfigLoader().get_embedding_config().emb_model_name) #AutoTokenizer.from_pretrained(emb_model)  # Tokenizer for the embedding model
        self.gen_tok = AutoTokenizer.from_pretrained(gen_tok_name) # Tokenizer for the generation model
        # Initialize the summarizer with GPU support
        self.summarizer = pipeline("summarization", 
                                   model="facebook/bart-large-cnn", 
                                   device=0 if self.device.type == "cuda" else -1)
        if isinstance(self.emb_model.max_seq_length, property):
            self.max_seq_len = self.emb_model.max_seq_length.fget(self.emb_model)
        else:
            self.max_seq_len = self.emb_model.max_seq_length
        self.chunk_size = min(chunk_size, self.max_seq_len -1) # reserve 1 token for safety
        self.overlap = overlap
        self.gen_context_window = ConfigLoader().get_generator_config().context_window
        self.store_tokens = store_tokens
        logger.info(f"Initialized RAGChunker with chunk size {self.chunk_size}, overlap {self.overlap}, and store_tokens {self.store_tokens}")

    def _tokenize_and_count(self, text: str) -> Union[int, tuple]:
        tokens = self.emb_tok.tokenize(text)
        if self.store_tokens:
            return len(tokens), tokens
        return len(tokens)

    def chunk_by_tokens(self, text: str) -> List[Union[str, Dict[str, Union[str, List[str]]]]]:
        """
        This method splits the text into chunks based on the token count. It uses overlap to ensure that the chunks are not cut in the middle of words.

        """
        chunks = []   # An empty list to store the chunks
        start_char = 0  # The starting index of the current chunk
        text_length = len(text)  # The length of the input text
        
        while start_char < text_length: # the loop continues until the 'start_char' reaches the end of the text ('text_length')
            end_char = min(start_char + 1, text_length) # Always move forward by at least one character, ensurin each chunk is at least one character long
            
            # Attempt to expand the current chunk by one character at a time, until the token count exceeds the chunk size (self.chunk_size)
            while end_char < text_length:
                next_end = min(end_char + 1, text_length)
                chunk_text = text[start_char:next_end]   
                tokens = self.emb_tok.tokenize(chunk_text)
                
                if len(tokens) > self.chunk_size: # If the token count exceeds the chunk size, break the loop
                    break
                
                end_char = next_end

            final_chunk_text = text[start_char:end_char] # the final chunk text is the text from 'start_char' to 'next_end'
            tokens = self.emb_tok.tokenize(final_chunk_text)

            #logger.info(f"Chunk {len(chunks)}: token count = {len(tokens)}, text = {final_chunk_text[:50]}...")

            if self.store_tokens:  # if 'store_tokens' is True, store the tokens along with the chunk text in a dictionary
                chunks.append({
                    'text': final_chunk_text,
                    'tokens': tokens
                })
            else:  # else store only the chunk text
                chunks.append(final_chunk_text)

            # Move start_char forward, considering overlap
            start_char = max(end_char - self.overlap, start_char + 1)  # If the overlap is 0, the start_char index is simply incremented by 1. Otherwise, it is incremented by the overlap value.

        return chunks
    
    def chunk_by_sentence(self, text: str) -> List[Union[str, Dict[str, Union[str, List[str]]]]]:
        """
        create chunks of text based on sentence boundaries, so that the chunks are manageable for the embedding model.
        The text is split into sentences using a simple period-based sentence splitting.
        
        Note that this implementation assumes that sentences are separated by periods (.) and may not work correctly for more complex sentence structures or punctuation.
        """
        #logger.info(f"Chunking text by sentences")
        sentences = text.split('.')  # Simple sentence splitting; sentences are assumed to be separated by periods
        chunks = []  # An empty list to store the chunks
        current_chunk = [] # An empty list to store the text of the current chunk
        current_tokens = [] # An empty list to store the tokens of the current chunk
        current_token_count = 0  # The token count of the current chunk

        for sentence in sentences:
            sentence = sentence.strip() + '.'  # Each sentence is stripped of leading/trailing whitespace and appended with a period (.) to ensure proper sentence structure.
            # Create tokens for the sentence and count them (the Embedding model's tokenizer is used here)
            sentence_tokens = self.emb_tok.tokenize(sentence) # Tokenize the sentence using the embedding model's tokenizer
            sentence_token_count = len(sentence_tokens) # The token count of the sentence
            if current_token_count + sentence_token_count <= self.chunk_size: #  If the total token count of the current chunk (current_token_count) plus the token count of the current sentence (sentence_token_count) does not exceed the maximum chunk size (self.chunk_size), the sentence is added to the current chunk.
                current_chunk.append(sentence) # the text of the sentence
                current_tokens.extend(sentence_tokens)  # the tokenized form of the sentence
                current_token_count += sentence_token_count # the token count of the sentence
            else:
                if self.store_tokens:
                    # Store the chunk text and tokens as we are done with the current chunk
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'tokens': current_tokens
                    })
                else:
                    chunks.append(' '.join(current_chunk))
                # Start a new chunk with the current sentence
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                current_token_count = sentence_token_count
        if current_chunk:
            if self.store_tokens:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'tokens': current_tokens
                })
            else:
                chunks.append(' '.join(current_chunk))
        #logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks
    

    def chunk_by_tokens_for_hybrid(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Chunk text by tokens with a specified chunk size and overlap, respecting the max sequence length.
        
        Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk in tokens.
        overlap (int): The number of overlapping tokens between chunks.
        
        Returns:
        List[str]: A list of text chunks.
        """
        chunks = []
        
      
        # Process the text in manageable segments
        segment_size = self.chunk_size * 10  # Process 10 chunks worth of text at a time
        segments = [text[i:i+segment_size] for i in range(0, len(text), segment_size)]
        
        current_chunk = []
        current_chunk_size = 0
        
        for segment in segments:
            tokens = self.emb_tok.tokenize(segment)
            
            i = 0
            while i < len(tokens):
                if current_chunk_size < self.chunk_size:
                    current_chunk.append(tokens[i])
                    current_chunk_size += 1
                    i += 1
                else:
                    # Find a good breaking point
                    break_point = i
                    for j in range(i, max(i - self.chunk_size // 10, 0), -1):
                        if tokens[j].endswith(('.', '!', '?', ',', ';')):
                            break_point = j + 1
                            break
                    
                    # Add the chunk
                    chunk_text = self.emb_tok.convert_tokens_to_string(current_chunk[:break_point-i+self.chunk_size])
                    chunks.append(chunk_text)
                    
                    # Start the next chunk with overlap
                    start_next_chunk = max(0, break_point - overlap)
                    current_chunk = tokens[start_next_chunk:break_point]
                    current_chunk_size = len(current_chunk)
                    i = break_point
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(self.emb_tok.convert_tokens_to_string(current_chunk))
        
        return chunks
    
    def chunk_by_tokens_for_hybrid2(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Helper method to chunk text by tokens with a specified chunk size and overlap.
        """
        chunks = []
        start = 0
        text_tokens = self.emb_tok.tokenize(text)
        
        while start < len(text_tokens):
            end = start + chunk_size
            chunk_tokens = text_tokens[start:end]
            chunk = self.emb_tok.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk)
            start += (chunk_size - overlap)
        
        return chunks

    def hybrid_chunk_document(self, text: str, 
                              large_chunk_size: int = 1000,
                              small_chunk_size: int = 200,
                              overlap: int = 50) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Implement a hybrid chunking approach with hierarchical chunking, semantic compression, and chunk linking.
        
        :param document: The input document to be chunked
        :param large_chunk_size: The target size for large chunks (in tokens)
        :param small_chunk_size: The target size for small chunks (in tokens)
        :param overlap: The number of overlapping tokens between chunks
        :return: A list of chunked documents with metadata
        """
        # Step 1: Create large chunks
        large_chunks = self.chunk_by_tokens_for_hybrid(text, chunk_size=large_chunk_size, overlap=overlap)
        #logger.info(f"Created {len(large_chunks)} large chunks")

        # Step 2: Process large chunks
        chunked_documents = []
        # Clear CUDA cache
        torch.cuda.empty_cache()

        summarizer = pipeline("summarization", 
                              model="facebook/bart-base", 
                              device=0 if self.device.type == "cuda" else -1)

        for i, large_chunk in enumerate(large_chunks):
            # Semantic compression
            summary = summarizer(large_chunk, max_length=50, min_length=30, do_sample=False)[0]['summary_text']

            # Create small chunks from the compressed large chunk
            small_chunks = self.chunk_by_tokens_for_hybrid(large_chunk, chunk_size=small_chunk_size, overlap=overlap)

            for j, small_chunk in enumerate(small_chunks):
                chunked_doc = text.copy()
                chunked_doc['text'] = small_chunk
                chunked_doc['large_chunk_id'] = i
                chunked_doc['small_chunk_id'] = j
                chunked_doc['summary'] = summary
                
                chunked_doc['metadata'] = {
                    'prev_chunk_id': j - 1 if j > 0 else None,
                    'next_chunk_id': j + 1 if j < len(small_chunks) - 1 else None,
                    'parent_chunk_id': i
                }

                # Calculate semantic importance (example: using token count as a simple proxy TODO: Improv this to something more sophisticated)
                chunked_doc['semantic_importance'] = len(self.emb_tok.tokenize(small_chunk))

                chunked_documents.append(chunked_doc)
            return chunked_documents

    def chunk_document(self, document: Dict[str, str], method: str = 'tokens') -> List[Dict[str, Union[str, List[str]]]]:
        #logger.info(f"Chunking document with method '{method}'")
        text = document['text']
        if method == 'tokens':
            text_chunks = self.chunk_by_tokens(text)
        elif method == 'sentence':
            text_chunks = self.chunk_by_sentence(text)
        elif method == 'hybrid':
            text_chunks = self.hybrid_chunk_document(text)
        else:
            raise ValueError("Invalid chunking method. Choose 'tokens', 'sentence' or 'hybrid'.")

        chunked_documents = []
        for i, chunk in enumerate(text_chunks):
            chunked_doc = document.copy()
            if isinstance(chunk, dict):
                chunked_doc.update(chunk)
            else:
                chunked_doc['text'] = chunk
            chunked_doc['chunk_id'] = i
            chunked_documents.append(chunked_doc)

        return chunked_documents
    
    def get_embeddings(self, chunks: List[Union[str, Dict[str, Union[str, List[str]]]]]) -> List[List[float]]:
        texts = [chunk['text'] if isinstance(chunk, dict) else chunk for chunk in chunks]
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def prepare_for_generation(self, query: str, relevant_chunks: List[Union[str, Dict[str, Union[str, List[str]]]]], system_prompt: str = "") -> str:
        generation_tokens = self.gen_tok.tokenize(query + system_prompt)
        available_tokens = self.generation_context_window - len(generation_tokens) - 100  # 100 tokens safety margin

        context = ""
        for chunk in relevant_chunks:
            chunk_text = chunk['text'] if isinstance(chunk, dict) else chunk
            chunk_tokens = self.gen_tok.tokenize(chunk_text)
            if len(chunk_tokens) + len(self.gen_tok.tokenize(context)) > available_tokens:
                break
            context += chunk_text + " "

        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"