from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from sentence_transformers import util

from config.config import ConfigLoader
from typing import List, Dict
from utils.logger.logging_config import logger
import torch

class Generator:
    def __init__(self, max_length: int = 512):
        #self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader().get_generator_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.gen_tokenizer)
        # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        self.model = AutoModelForCausalLM.from_pretrained(
                self.config.gen_model_name,
                model_file=self.config.gen_model_file,
                model_type=self.config.gen_model_type,
                gpu_layers=self.config.gpu_layers
            )
        self.max_length = max_length
        self.chat_history: List[Dict[str, str]] = []  # a list of dictionaries where both the keys and values are strings (role:message)
        logger.info(f"Initialized Generator: context length = {format(self.model.config.context_length)}")

    # Append a message to the chat history
    def add_msg(self, role: str, message: str):
        self.chat_history.append({"role": role, "content": message})

    def generate(self, query: str, context: str) -> str:

        # Construct a more structured prompt; longer prompts are seen to take a lot of time to generate
        system_message = """You are LegalGenie, an AI legal assistant. Your task is to provide accurate information based solely on the given context. Follow these rules strictly:
1. Only use information explicitly stated in the context.
2. Do not add opinions, suggestions, or information not present in the context.
3. Do not provide legal advice.
4. If the context doesn't contain relevant information to answer the query, state that you don't have enough information to answer accurately.
5. Cite the source of information when possible, using the format (Source X) where X is the source number.
"""

        system_message = "You are LegalGenie, an AI legal assistant. Provide only factual information directly from the given context. Do not add opinions, suggestions, or information not explicitly stated in the context. Do not provide legal advice. Do not invent facts not present in the context. Do not provide information that is not supported by the context. Do not provide information that is not relevant to the context. Do not provide information that is not directly related to the query"
        
        #context_prompt = f"Relevant legal context: {context}\n\nUse this information to inform your response, but do not quote it directly unless necessary."
        context_prompt = f"Context:\n{context}\n\nUse this information to answer the query. Cite sources when possible."
        
        human_message = f"Human: {query}"
        
        ai_prompt = "LegalGenie: Based on the provided legal context and the query, here's my response:"

        # Combine all parts into a single prompt
        full_prompt = f"{system_message}\n\n{context_prompt}\n\n{human_message}\n\n{ai_prompt}"

        # Generate the output
        outputs = self.model(
            full_prompt,
            max_new_tokens=self.max_length, # Increase for more comprehensive responses
            temperature=0.1, # Lower temperature for more deterministic responses; helps stick to the legal context
            top_k=50,
            top_p=0.95
        )

        # Extract the generated response (everything after the last occurrence of "LegalGenie:")       
        response = outputs.split("LegalGenie:")[-1].strip()

        #if not self.verify_response(response, context):
        #    response += "\n\nNote: This response may not be fully grounded in the provided context. Please verify the information."
        
        return response


        # Update chat history
        #self.chat_history.add_msg("human", query)
        #self.chat_history.add_msg("ai", response)
        
        return response

    def verify_response(self, response: str, context: str) -> bool:
        response_embedding = self.model.encode(response, convert_to_tensor=True)
        context_embedding = self.model.encode(context, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(response_embedding, context_embedding)
        
        return similarity.item() > 0.5  # Adjust this threshold as needed
    
    # Get the chat history
    def get_chat_history(self) -> List[Dict[str, str]]:
        return self.chat_history

    # Clear the chat history
    def clear_chat_history(self):
        self.chat_history.clear()