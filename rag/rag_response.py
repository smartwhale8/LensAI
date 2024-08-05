# File: rag_response.py

from typing import List, Dict
import json

class RAGResponse:
    def __init__(self, generated_text: str, similar_docs: List[Dict]):
        self.generated_text = generated_text
        self.similar_docs = similar_docs

    def get_formatted_response(self) -> str:
        response = self.generated_text + "\n\nSources:\n"
        for i, doc in enumerate(self.similar_docs, 1):
            metadata = doc['metadata']
            response += f"{i}. {metadata.get('short_title', 'Untitled')} (Similarity: {metadata['similarity_score']:.2f})\n"
            response += f"   ID: {metadata.get(next(iter(metadata)), 'Unknown')}\n"
            response += f"   Excerpt: {doc['relevant_excerpt'][:200]}...\n\n"
        return response

    def get_metadata(self) -> Dict:
        return {
            "similar_documents": [
                {
                    "id": doc['metadata'].get(next(iter(doc['metadata'])), 'Unknown'),
                    "similarity_score": doc['metadata']['similarity_score'],
                    "short_title": doc['metadata'].get('short_title', 'Untitled'),
                    "long_title": doc['metadata'].get('long_title', 'No long title available'),
                    "year": doc['metadata'].get('act_year', 'Unknown year'),
                    "excerpt_length": len(doc['relevant_excerpt']),
                    "excerpt_preview": doc['relevant_excerpt'][:100] + "..."
                }
                for doc in self.similar_docs
            ],
            "total_documents": len(self.similar_docs)
        }

    def get_raw_generated_text(self) -> str:
        return self.generated_text

    def get_similar_docs(self) -> List[Dict]:
        return self.similar_docs

    def add_similar_doc(self, doc: Dict) -> None:
        self.similar_docs.append(doc)

    def set_generated_text(self, text: str) -> None:
        self.generated_text = text

    def get_json_response(self) -> str:
        return json.dumps({
            "generated_text": self.generated_text,
            "metadata": self.get_metadata()
        }, indent=2)

    def __str__(self) -> str:
        return self.get_formatted_response()

    def __repr__(self) -> str:
        return f"RAGResponse(generated_text='{self.generated_text[:50]}...', similar_docs={len(self.similar_docs)} docs)"