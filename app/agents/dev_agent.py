import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DevAgent(BaseAgent):
    def __init__(self, llm_manager, config: Dict[str, Any] = None):
        super().__init__(llm_manager, config)
        self.name = "Development Agent"
        self.role = "Write, debug, and optimize code"
        
    def act(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"""You are a Senior Software Developer. Your role is to write high-quality code.

TASK: {task}

Please provide:
1. Clean, efficient code
2. Explanation of the solution
3. Time and space complexity analysis
4. Edge cases considered

Response:"""
        response = self.llm_manager.generate(prompt)
        
        return {
            "agent": self.name,
            "task": task,
            "response": response,
            "status": "completed"
        }
