import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from chromadb import ClientAPI
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages long-term memory for agents using vector storage"""
    
    def __init__(self, vector_store: VectorStore, collection_name: str = "agent_memory"):
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.collection = self.vector_store.get_or_create_collection(collection_name)
        
    def store_memory(self, 
                    content: str, 
                    agent_name: str,
                    memory_type: str = "general",
                    metadata: Dict[str, Any] = None) -> str:
        """Store a memory with metadata"""
        try:
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            enhanced_metadata = {
                "memory_id": memory_id,
                "agent_name": agent_name,
                "memory_type": memory_type,
                "timestamp": timestamp,
                "content_length": len(content)
            }
            
            if metadata:
                enhanced_metadata.update(metadata)
            
            self.vector_store.add_documents(
                collection_name=self.collection_name,
                documents=[content],
                metadatas=[enhanced_metadata],
                ids=[memory_id]
            )
            
            logger.info(f"Stored memory for {agent_name}, type: {memory_type}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def retrieve_memories(self, 
                         query: str, 
                         agent_name: Optional[str] = None,
                         memory_type: Optional[str] = None,
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query and filters"""
        try:
            # Build metadata filter
            filter_dict = {}
            if agent_name:
                filter_dict["agent_name"] = agent_name
            if memory_type:
                filter_dict["memory_type"] = memory_type
            
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_texts=[query],
                n_results=limit,
                where=filter_dict if filter_dict else None
            )
            
            memories = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    memory_data = {
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else None
                    }
                    memories.append(memory_data)
            
            logger.info(f"Retrieved {len(memories)} memories for query: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []
    
    def get_agent_context(self, agent_name: str, current_task: str, limit: int = 3) -> str:
        """Get relevant context for an agent's current task"""
        memories = self.retrieve_memories(
            query=current_task,
            agent_name=agent_name,
            limit=limit
        )
        
        if not memories:
            return "No relevant previous memories found."
        
        context = "Previous relevant memories:\n"
        for i, memory in enumerate(memories, 1):
            context += f"{i}. {memory['content']}\n"
            context += f"   (From: {memory['metadata'].get('timestamp', 'unknown')})\n\n"
        
        return context
    
    def clear_agent_memories(self, agent_name: str) -> int:
        """Clear all memories for a specific agent"""
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_texts=[""],
                n_results=1000,  # Large number to get all
                where={"agent_name": agent_name}
            )
            
            if results and results['ids']:
                self.vector_store.delete_documents(
                    collection_name=self.collection_name,
                    ids=results['ids'][0]
                )
                logger.info(f"Cleared {len(results['ids'][0])} memories for {agent_name}")
                return len(results['ids'][0])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing memories for {agent_name}: {str(e)}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_texts=[""],
                n_results=10000  # Get all for stats
            )
            
            if not results or not results['metadatas']:
                return {"total_memories": 0, "agents": {}}
            
            stats = {
                "total_memories": len(results['metadatas'][0]),
                "agents": {},
                "memory_types": {}
            }
            
            for metadata in results['metadatas'][0]:
                agent_name = metadata.get('agent_name', 'unknown')
                memory_type = metadata.get('memory_type', 'unknown')
                
                # Count by agent
                if agent_name not in stats['agents']:
                    stats['agents'][agent_name] = 0
                stats['agents'][agent_name] += 1
                
                # Count by memory type
                if memory_type not in stats['memory_types']:
                    stats['memory_types'][memory_type] = 0
                stats['memory_types'][memory_type] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}