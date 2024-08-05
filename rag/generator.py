from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from config.config import ConfigLoader
from typing import List, Dict
import logging
import torch

class Generator:
    def __init__(self, max_length: int = 512):
        self.logger = logging.getLogger(__name__)
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

    # Append a message to the chat history
    def add_msg(self, role: str, message: str):
        self.chat_history.append({"role": role, "content": message})

    def generate(self, query: str, context: str) -> str:

        # Construct a more structured prompt; longer prompts are seen to take a lot of time to generate
        system_message = "You are LegalGenie, an AI legal assistant. Provide only factual information directly from the given context. Do not add opinions, suggestions, or information not explicitly stated in the context."
        
        context_prompt = f"Relevant legal context: {context[:500]}\n\nUse this information to inform your response, but do not quote it directly unless necessary."
        
        human_message = f"Human: {query}"
        
        ai_prompt = "LegalGenie: Based on the provided legal context and the query, here's my response:"

        # Combine all parts into a single prompt
        full_prompt = f"{system_message}\n\n{context_prompt}\n\n{human_message}\n\n{ai_prompt}"

        # Tokenize the input
        #inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate the output
        with torch.no_grad():
            outputs = self.model(
                full_prompt,
                max_new_tokens=self.max_length, # Increase for more comprehensive responses
                temperature=0.3, # Lower temperature for more deterministic responses; helps stick to the legal context
                top_k=50,
                top_p=0.95
            )
        
        # Decode the generated text
        #full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Consume the generator to get the full output
        full_output = full_prompt + ''.join(outputs)

        # Extract the generated response
        response = full_output[len(full_prompt):].strip()
        
        # Update chat history
        #self.chat_history.add_msg("human", query)
        #self.chat_history.add_msg("ai", response)
        
        return response

    # Get the chat history
    def get_chat_history(self) -> List[Dict[str, str]]:
        return self.chat_history

    # Clear the chat history
    def clear_chat_history(self):
        self.chat_history.clear()