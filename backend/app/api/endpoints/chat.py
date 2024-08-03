from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from rag.database.mongodb_handler import MongoDBHandler
from rag.database.milvus_handler import MilvusHandler
from rag.embedding.embedding_handler import EmbeddingHandler
from rag.retriever import Retriever, CollectionConfigFactory
from rag.generator import Generator

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Initialize RAG components
mongodb_handler = MongoDBHandler(connection_string="mongodb://localhost:27017/", db_name="rag_lens_ai")
milvus_handler = MilvusHandler(host='localhost', port='19530')
embedding_handler = EmbeddingHandler("sentence-transformers/all-mpnet-base-v2", mongodb_handler, milvus_handler)
retriever = Retriever(
    milvus_handler, 
    mongodb_handler, 
    embedding_handler, 
    CollectionConfigFactory.get_config("legal_acts")
)
generator = Generator()

chat_history: List[Tuple[str, str]] = []

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_query = request.query

        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(query_text=user_query)

        if not retrieved_docs:
            return ChatResponse(response="I'm sorry, I couldn't find any relevant documents based on your query.")
        
        context = retrieved_docs[0]["relevant_excerpt"]  # Use the most similar document as context

        # Generate response
        response = generator.generate(user_query, context)

        chat_history.append(("User", user_query))
        chat_history.append(("LegalGenie", response))

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat_history")
async def get_chat_history():
    return {"history": chat_history}