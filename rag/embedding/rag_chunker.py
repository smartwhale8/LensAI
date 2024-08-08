from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from config.config import ConfigLoader
from typing import List, Dict, Union
from utils.logger.logging_config import logger
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
        self.emb_model = emb_model
        self.emb_tok = AutoTokenizer.from_pretrained(ConfigLoader().get_embedding_config().emb_model_name) #AutoTokenizer.from_pretrained(emb_model)  # Tokenizer for the embedding model
        self.gen_tok = AutoTokenizer.from_pretrained(gen_tok_name) # Tokenizer for the generation model
        self.chunk_size = min(chunk_size, self.emb_model.max_seq_length -1) # reserve 1 token for safety
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
        chunks = []
        start_char = 0
        text_length = len(text)
        
        while start_char < text_length:
            # Always move forward by at least one character
            end_char = min(start_char + 1, text_length)
            
            # Try to expand the chunk up to the maximum allowed size
            while end_char < text_length:
                next_end = min(end_char + 1, text_length)
                chunk_text = text[start_char:next_end]
                tokens = self.emb_tok.tokenize(chunk_text)
                
                if len(tokens) > self.chunk_size:
                    break
                
                end_char = next_end

            final_chunk_text = text[start_char:end_char]
            tokens = self.emb_tok.tokenize(final_chunk_text)

            #logger.info(f"Chunk {len(chunks)}: token count = {len(tokens)}, text = {final_chunk_text[:50]}...")

            if self.store_tokens:
                chunks.append({
                    'text': final_chunk_text,
                    'tokens': tokens
                })
            else:
                chunks.append(final_chunk_text)

            # Move start_char forward, considering overlap
            start_char = max(end_char - self.overlap, start_char + 1)

        return chunks
    
    def chunk_by_sentence(self, text: str) -> List[Union[str, Dict[str, Union[str, List[str]]]]]:
        """
        create chunks of text that are managable for the embedding model.
        Splits the text into chunks based on sentence boundaries.
        """
        #logger.info(f"Chunking text by sentences")
        sentences = text.split('.')  # Simple sentence splitting; sentences are assumed to be separated by periods
        chunks = []
        current_chunk = []
        current_tokens = []
        current_token_count = 0

        for sentence in sentences:
            sentence = sentence.strip() + '.'
            # Create tokens for the sentence and count them (the Embedding model's tokenizer is used here)
            sentence_tokens = self.emb_tok.tokenize(sentence)
            sentence_token_count = len(sentence_tokens)
            if current_token_count + sentence_token_count <= self.chunk_size:
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
    
    def chunk_document(self, document: Dict[str, str], method: str = 'tokens') -> List[Dict[str, Union[str, List[str]]]]:
        #logger.info(f"Chunking document with method '{method}'")
        text = document['text']
        if method == 'tokens':
            text_chunks = self.chunk_by_tokens(text)
        elif method == 'sentence':
            text_chunks = self.chunk_by_sentence(text)
        else:
            raise ValueError("Invalid chunking method. Choose 'tokens' or 'sentence'.")

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