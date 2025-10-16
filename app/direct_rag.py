"""
Direct RAG system that processes documents on-the-fly without storage
"""

import logging
from typing import Dict, Any, List, Optional
from app.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class DirectRAG:
    """Direct RAG system that works with documents without storing them"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        logger.info("DirectRAG initialized - no document storage")
    
    def process_and_query(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Process uploaded documents and answer query directly
        
        Args:
            documents: List of document dicts with 'content', 'filename', 'file_type'
            query: User's question
        """
        try:
            if not documents:
                return {
                    "response": "No documents provided. Please upload documents to query.",
                    "query": query,
                    "documents_processed": 0,
                    "has_context": False
                }
            
            # Build context from all provided documents
            context_parts = []
            doc_info = []
            
            for i, doc in enumerate(documents, 1):
                content = doc.get('content', '')
                filename = doc.get('filename', f'document_{i}')
                file_type = doc.get('file_type', 'unknown')
                
                if content.strip():
                    context_parts.append(f"Document {i} ({filename}):\n{content}")
                    doc_info.append({
                        "filename": filename,
                        "file_type": file_type,
                        "content_length": len(content),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    })
            
            if not context_parts:
                return {
                    "response": "No valid content found in the uploaded documents.",
                    "query": query,
                    "documents_processed": len(documents),
                    "has_context": False
                }
            
            # Build prompt with all document context
            context = "\n\n".join(context_parts)
            prompt = f"""
You are a helpful assistant. Use the provided documents to answer the user's question accurately.

DOCUMENTS:
{context}

USER QUESTION: {query}

Instructions:
1. Base your answer primarily on the provided documents
2. Use specific information from the documents to answer the question
3. If the documents contain relevant information, cite which document(s) you're referencing
4. If the documents don't contain relevant information for the question, say so clearly
5. Be comprehensive and helpful in your response

Answer:"""
            
            # Generate response
            response = self.llm_manager.generate(prompt)
            
            return {
                "response": response,
                "query": query,
                "documents_processed": len(doc_info),
                "processed_documents": doc_info,
                "has_context": True
            }
            
        except Exception as e:
            logger.error(f"Error in direct RAG processing: {str(e)}")
            return {
                "response": f"Error processing documents: {str(e)}",
                "query": query,
                "documents_processed": 0,
                "processed_documents": [],
                "has_context": False,
                "error": str(e)
            }
    
    def query_single_document(self, content: str, filename: str, file_type: str, query: str) -> Dict[str, Any]:
        """
        Process a single document and answer query
        """
        document = {
            "content": content,
            "filename": filename,
            "file_type": file_type
        }
        return self.process_and_query([document], query)
