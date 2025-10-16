import logging
import json
import yaml
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class UtilityHelper:
    """Utility functions for the multi-agent system"""
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """Generate unique ID"""
        unique_id = str(uuid.uuid4())
        return f"{prefix}_{unique_id}" if prefix else unique_id
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file"""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            else:  # Assume JSON
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:  # Assume JSON
                with open(path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            raise
    
    @staticmethod
    def format_agent_response(response: Dict[str, Any]) -> str:
        """Format agent response for display"""
        formatted = f"Agent: {response.get('agent', 'Unknown')}\n"
        formatted += f"Task: {response.get('task', 'Unknown')}\n"
        formatted += f"Status: {response.get('status', 'Unknown')}\n"
        formatted += f"Response: {response.get('response', 'No response')}\n"
        
        # Add additional fields
        for key, value in response.items():
            if key not in ['agent', 'task', 'status', 'response']:
                if isinstance(value, (list, dict)) and value:
                    formatted += f"{key.title()}: {json.dumps(value, indent=2)}\n"
                elif value:
                    formatted += f"{key.title()}: {value}\n"
        
        return formatted
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def validate_file_type(file_path: str, allowed_types: List[str]) -> bool:
        """Validate file type"""
        path = Path(file_path)
        file_extension = path.suffix.lower().lstrip('.')
        return file_extension in allowed_types
    
    @staticmethod
    def create_directory(dir_path: str) -> bool:
        """Create directory if it doesn't exist"""
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            return False
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity (cosine similarity approximation)"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0