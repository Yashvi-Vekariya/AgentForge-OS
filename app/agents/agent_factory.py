import logging
from typing import Dict, Any
from app.llm_manager import LLMManager
from .base_agent import BaseAgent
from .dev_agent import DevAgent
from .research_agent import ResearchAgent
from .vision_agent import VisionAgent
from .data_agent import DataAgent
from .product_agent import ProductAgent
from .design_agent import DesignAgent

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory class for creating different types of agents"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_registry = {
            "dev": DevAgent,
            "research": ResearchAgent,
            "vision": VisionAgent,
            "data": DataAgent,
            "product": ProductAgent,
            "design": DesignAgent
        }
    
    def create_agent(self, agent_type: str, config: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent of specified type"""
        if agent_type not in self.agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self.agent_registry.keys())}")
        
        agent_class = self.agent_registry[agent_type]
        return agent_class(self.llm_manager, config or {})
    
    def get_available_agents(self) -> list:
        """Get list of available agent types"""
        return list(self.agent_registry.keys())
