import logging
import re
from typing import Dict, Any, List, Optional
from app.vector_store import VectorStore
from app.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine for enhanced responses"""
    def __init__(self, vector_store: VectorStore, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.collection_name = "rag_documents"
        
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add documents to RAG collection"""
        try:
            if metadatas is None:
                import time
                # Create non-empty metadata for each document
                metadatas = [
                    {
                        "document_index": i,
                        "added_timestamp": time.time(),
                        "source": "rag_engine"
                    }
                    for i, _ in enumerate(documents)
                ]
            else:
                # Ensure all metadata dictionaries are non-empty
                for i, metadata in enumerate(metadatas):
                    if not metadata:  # Empty dict
                        metadatas[i] = {
                            "document_index": i,
                            "added_timestamp": time.time(),
                            "source": "rag_engine"
                        }

            document_ids = self.vector_store.add_documents(
                collection_name=self.collection_name,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(document_ids)} documents to RAG collection")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to RAG: {str(e)}")
            raise
    
    def query(self, 
             query: str, 
             n_results: int = 3,
             filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query RAG system and return relevant documents"""
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            if not results or not results['documents']:
                return {
                    "query": query,
                    "documents": [],
                    "metadatas": [],
                    "distances": []
                }
            
            return {
                "query": query,
                "documents": results['documents'][0],
                "metadatas": results['metadatas'][0],
                "distances": results['distances'][0] if results['distances'] else []
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG: {str(e)}")
            return {
                "query": query,
                "documents": [],
                "metadatas": [],
                "distances": [],
                "error": str(e)
            }
    
    def generate_with_rag(self, 
                         query: str, 
                         context: Optional[str] = None,
                         n_documents: int = 3) -> Dict[str, Any]:
        """Generate response using RAG-enhanced context"""
        try:
            # Retrieve relevant documents
            rag_results = self.query(query, n_results=n_documents)
            retrieved_docs = rag_results['documents']
            
            # Build enhanced prompt
            prompt = self._build_rag_prompt(query, retrieved_docs, context)
            
            # Generate response
            response = self.llm_manager.generate(prompt)
            
            return {
                "response": response,
                "retrieved_documents": retrieved_docs,
                "document_metadatas": rag_results['metadatas'],
                "query": query,
                "rag_used": len(retrieved_docs) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "retrieved_documents": [],
                "document_metadatas": [],
                "query": query,
                "rag_used": False,
                "error": str(e)
            }
    
    def _build_rag_prompt(self, query: str, documents: List[str], context: Optional[str]) -> str:
        """Build prompt with RAG context"""
        documents_text = ""
        if documents:
            documents_text = "REFERENCE DOCUMENTS:\n"
            for i, doc in enumerate(documents, 1):
                documents_text += f"Document {i}: {doc}\n\n"
        
        context_text = f"ADDITIONAL CONTEXT: {context}\n\n" if context else ""
        
        prompt = f"""
        You are a knowledgeable assistant. Use the provided reference documents and context to answer the user's query accurately.
        
        {documents_text}
        {context_text}
        USER QUERY: {query}
        
        Instructions:
        1. Base your answer primarily on the reference documents provided
        2. If the documents don't contain relevant information, use your general knowledge but indicate this
        3. Be precise and cite which document(s) you're referencing when applicable
        4. If the query requires information not in the documents, acknowledge this limitation
        
        Provide a comprehensive, well-structured response:
        """
        
        return prompt
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on stored documents"""
        rag_results = self.query(query, n_results=n_results)
        
        results = []
        for i, (doc, metadata) in enumerate(zip(rag_results['documents'], rag_results['metadatas'])):
            result = {
                "content": doc,
                "metadata": metadata,
                "rank": i + 1,
                "distance": rag_results['distances'][i] if rag_results['distances'] else None
            }
            results.append(result)
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about RAG collection"""
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_texts=[""],
                n_results=1
            )
            
            total_docs = 0
            if results and results['ids']:
                total_docs = len(results['ids'][0])
            
            return {
                "collection_name": self.collection_name,
                "total_documents": total_docs,
                "status": "active" if total_docs > 0 else "empty"
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG stats: {str(e)}")
            return {"error": str(e)}