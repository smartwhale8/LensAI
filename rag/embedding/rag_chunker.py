from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from config.config import ConfigLoader
from typing import List, Dict, Union
import re

class RAGChunker:
    def __init__(self, 
            emb_model_name = ConfigLoader().get_embedding_config().emb_model_name,
            gen_model_name = ConfigLoader().get_generator_config().gen_model_name, 
            chunk_size: int = 3500, 
            overlap: int = 50,
            store_tokens: bool = False):
        self.emb_model = SentenceTransformer(emb_model_name)
        self.gen_model = AutoTokenizer.from_pretrained(gen_model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.gen_context_window = ConfigLoader.get_generator_config().context_window
        self.store_tokens = store_tokens

    def _tokenize_and_count(self, text: str) -> Union[int, tuple]:
        tokens = self.emb_model.tokenizer.tokenize(text)
        if self.store_tokens:
            return len(tokens), tokens
        return len(tokens)

    def chunk_by_tokens(self, text: str) -> List[Union[str, Dict[str, Union[str, List[str]]]]]:
        token_count, tokens = self._tokenize_and_count(text) if self.store_tokens else (self._tokenize_and_count(text), None)
        chunks = []
        start = 0
        while start < token_count:
            end = start + self.chunk_size
            if end > token_count:
                end = token_count
            chunk_text = self.embedding_model.tokenizer.convert_tokens_to_string(tokens[start:end] if tokens else self.embedding_model.tokenizer.tokenize(text)[start:end])
            if self.store_tokens:
                chunks.append({
                    'text': chunk_text,
                    'tokens': tokens[start:end]
                })
            else:
                chunks.append(chunk_text)
            start = end - self.overlap
        return chunks
    
    def chunk_by_sentence(self, text: str) -> List[Union[str, Dict[str, Union[str, List[str]]]]]:
        sentences = text.split('.')  # Simple sentence splitting
        chunks = []
        current_chunk = []
        current_tokens = []
        current_token_count = 0
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_token_count, sentence_tokens = self._tokenize_and_count(sentence) if self.store_tokens else (self._tokenize_and_count(sentence), None)
            if current_token_count + sentence_token_count <= self.chunk_size:
                current_chunk.append(sentence)
                if self.store_tokens:
                    current_tokens.extend(sentence_tokens)
                current_token_count += sentence_token_count
            else:
                if self.store_tokens:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'tokens': current_tokens
                    })
                else:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens if self.store_tokens else []
                current_token_count = sentence_token_count
        if current_chunk:
            if self.store_tokens:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'tokens': current_tokens
                })
            else:
                chunks.append(' '.join(current_chunk))
        return chunks
    
    def chunk_document(self, document: Dict[str, str], method: str = 'tokens') -> List[Dict[str, Union[str, List[str]]]]:
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
        return self.embedding_model.encode(texts).tolist()

    def prepare_for_generation(self, query: str, relevant_chunks: List[Union[str, Dict[str, Union[str, List[str]]]]], system_prompt: str = "") -> str:
        generation_tokens = self.generation_tokenizer.tokenize(query + system_prompt)
        available_tokens = self.generation_context_window - len(generation_tokens) - 100  # 100 tokens safety margin

        context = ""
        for chunk in relevant_chunks:
            chunk_text = chunk['text'] if isinstance(chunk, dict) else chunk
            chunk_tokens = self.generation_tokenizer.tokenize(chunk_text)
            if len(chunk_tokens) + len(self.generation_tokenizer.tokenize(context)) > available_tokens:
                break
            context += chunk_text + " "

        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"