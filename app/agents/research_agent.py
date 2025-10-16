import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    def __init__(self, llm_manager, config: Dict[str, Any] = None):
        super().__init__(llm_manager, config)
        self.name = "Research Agent"
        self.role = "Research and analyze information"
        
    def act(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"""You are a Research Analyst. Your role is to provide comprehensive research.

TASK: {task}

Please provide:
1. Detailed analysis
2. Key findings
3. Sources and references
4. Recommendations

Response:"""
        response = self.llm_manager.generate(prompt)
        
        return {
            "agent": self.name,
            "task": task,
            "response": response,
            "sources": ["Research completed"],
            "status": "completed"
        }
