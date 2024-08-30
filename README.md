# LegalGenie LensAI
is a Retrieval-Augmented Generation (RAG) system purpose-built for legal research use-case. The system uses open-source components such as various Large Language Models, Vector Database etc.

The system aims to empower legal professionals to conduct highly efficient and accurate legal research across a comprehensive corpus of all legal acts passed by the Indian Central Government from 1836 to 2024. The data has been sourced from the official public website for legal acts of the Indian Government, indiacode.nic.in.
The legal acts passed the various state governments of India is out of scope of this project as of now.

## Terminology
* **Embeddings**: Dense vector representations of data, such as text or images, that capture their semantic meaning in a lower-dimensional space. These vectors are designed to position similar data points (e.g., similar words, sentences, or concepts) close to each other in this space. This proximity enables efficient similarity comparisons and facilitates tasks like information retrieval and clustering.
* **RAG**: A RAG system enhances the quality and accuracy of generated responses by integrating retrieval mechanisms with generative models. It works by first retrieving relevant information from an external knowledge source, using embeddings to search and match the most pertinent data. This retrieved content is then used to augment and guide the generative model (typically an LLM), allowing it to produce responses that are more contextually informed, accurate, and grounded in factual data. This method ensures that the generative model is not solely reliant on its pre-trained knowledge but can leverage up-to-date and context-specific information during inference.
* **Large Language Models (LLMs)**: Advanced neural network-based models trained on vast amounts of text data, enabling them to understand and generate human-like text. These models, often based on transformer architectures, excel at a wide range of natural language processing tasks, including language generation, translation, summarization, and question-answering. When used in a RAG system, LLMs are enhanced by the retrieval of relevant external data, making their outputs more reliable and context-aware.
  
## RAG Pipeline
1. Text Preprocessing
2. Embedding Model Selection: The Embedding model was chosen keeping in mind the limitation of the developent environment at hand. The LensAI system provides easy configuration based mechanism to swap out the various models to match the capabilities of the hardware and architecture. 
3. Genearating Embeddings: The pre-processed text is first converted to tokens using a tokenizer associated with the model. The text is passed through the embedding model to obtain a dense vector (embedding). The system allows the following 3 modes of embedding representations:
   * sentences (using periods as the sentence boundary).
   * tokens
   * hybrid (although this particular part is still a work in progress, the goal here is to achieve a hierarchical representation deriving naturally from the legal acts's structure, namely, sections, subsections, paragraph, sentences etc.)
5. Embedding Storage: Once the embeddings are created by the Embedding model, it is stored in Milvus to enable efficient similarity search during the retrieval phase of the RAG pipeline.
6. Retrieval Process in RAG
   * Query Embedding: The query is embedded using the same embedding model
   * Similarity Search: The query embedding is compared against the precomputed embeddings in the vector database. Common similarity metrics include cosine similarity or Euclidean distance. The most similar embeddings (and their corresponding documents) are retrieved
   * Document Retrieval: The corresponding passages are retrieved based on the most similar embeddings. 
7. Augmentation and Generation
   After retrieving the relevant documents, the RAG model uses them to augment the input for the generative model (like GPT). This augmented input helps generate a more accurate and contextually aware response.


## Components used
1. The vector storage comes from Milvus https://milvus.io/
2. The noSQL database, MongoDB https://www.mongodb.com/ is used for storing the non-vectorized components of the legal acts
3. The following LLMs are used for various stages in the RAG pipeline:
   - Embedding creation model (for queries and the vectorized data) : https://huggingface.co/sentence-transformers/all-mpnet-base-v2
   - Generation model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

## System Specifications
The system is developed to run locally on a desktop workstation, although with a large enough VRAM GPU to fit the various LLMs.
The developent PC was equipped with an Nvidia RTX 3060 (12GB VRAM).
Together with the 32 GB RAM it was easily able to satisfy running quanitized version of Llama2 7B LLM as well as Milvus Docker container which alone requires 8GB RAM.
It was powered by an AMD Ryzen 7 3700X 8-Core Processor.


It must be noted that although Embedding Models built on large-scale architectures like BERT, GPT, or RoBERTa can be called LLMs, there are several smaller or simpler models (e.g., Word2Vec, GloVe, FastText or lightweight sentence encoders) are more aptly referred simply as embedding models due to not meeting the definition of 'large' in an LLM.
