# LegalGenie LensAI
is a Retrieval Augmented Generation system purpose-built for legal research use-case. The system uses open-source components such as various Large Language Models, Vector Database etc.

1. The vector storage comes from Milvus https://milvus.io/
2. The noSQL database, MongoDB https://www.mongodb.com/ is used for storing the non-vectorized components of the legal acts
3. The following LLMs are used for various stages in the RAG pipeline:
   - Embedding creation model (for queries and the vectorized data) : https://huggingface.co/sentence-transformers/all-mpnet-base-v2
   - Generation model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

### System Requirements:
The system is developed to run locally on a desktop workstation, although with a large enough VRAM GPU to fit the various LLMs.
The developent PC was equipped with an Nvidia RTX 3060 (12GB VRAM), and together with the 32 GB RAM it was easily able to satisfy running quanitized version of Llama2 7B LLM as well as Milvus Docker container which alone requires 8GB RAM.
