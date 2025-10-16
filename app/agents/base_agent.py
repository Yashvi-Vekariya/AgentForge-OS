import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        self.llm_manager = llm_manager
        self.config = config or {}
        self.name = "Base Agent"
        self.role = "Generic agent role"
    
    @abstractmethod
    def act(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute agent task - to be implemented by subclasses"""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities"""
        return {
            "name": self.name,
            "role": self.role,
            "llm_config": self.llm_manager.get_config() if hasattr(self.llm_manager, 'get_config') else {}
        }
