import chromadb
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, path: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=path)
        logger.info(f"VectorStore initialized at: {path}")
    
    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(collection_name)
    
    def add_documents(self, 
                     collection_name: str, 
                     documents: List[str], 
                     metadatas: List[Dict] = None, 
                     ids: List[str] = None):
        """Add documents to collection"""
        collection = self.get_or_create_collection(collection_name)
        
        if metadatas is None:
            import time
            # Create non-empty metadata for each document
            metadatas = [
                {
                    "document_index": i,
                    "added_timestamp": time.time(),
                    "source": "vector_store"
                }
                for i, _ in enumerate(documents)
            ]
        else:
            # Ensure all metadata dictionaries are non-empty
            for i, metadata in enumerate(metadatas):
                if not metadata:  # Empty dict
                    import time
                    metadatas[i] = {
                        "document_index": i,
                        "added_timestamp": time.time(),
                        "source": "vector_store"
                    }
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return ids
    
    def query(self, 
              collection_name: str, 
              query_texts: List[str], 
              n_results: int = 5,
              where: Dict = None):
        """Query collection for similar documents"""
        collection = self.get_or_create_collection(collection_name)
        
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )
        return results
    
    def delete_documents(self, collection_name: str, ids: List[str]):
        """Delete documents from collection"""
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=ids)