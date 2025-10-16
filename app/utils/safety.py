import logging
import re
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

class SafetyFilter:
    """Safety filter for content moderation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.harmful_patterns = [
            r"(?i)(kill|harm|hurt|violence|attack|destroy)",
            r"(?i)(hate|racist|sexist|discriminat)",
            r"(?i)(illegal|crime|criminal)",
            r"(?i)(dangerous|weapon|gun|explosive)",
            # Add more patterns as needed
        ]
        
        self.sensitive_topics = [
            "self-harm", "suicide", "violence", "harassment",
            "discrimination", "illegal activities", "malware"
        ]
    
    def check_content_safety(self, text: str) -> Dict[str, Any]:
        """Check text for harmful content"""
        try:
            issues_found = []
            
            # Pattern matching
            for pattern in self.harmful_patterns:
                if re.search(pattern, text):
                    issues_found.append(f"Pattern matched: {pattern}")
            
            # Topic detection
            text_lower = text.lower()
            for topic in self.sensitive_topics:
                if topic.replace("-", "").replace(" ", "") in text_lower:
                    issues_found.append(f"Sensitive topic: {topic}")
            
            # Length-based heuristic (very short suspicious responses)
            if len(text.strip()) < 10 and any(keyword in text_lower for keyword in 
                ['yes', 'no', 'ok', 'sure', 'fine']):
                issues_found.append("Overly simplistic response")
            
            safety_score = max(0, 100 - len(issues_found) * 20)
            is_safe = len(issues_found) == 0
            
            return {
                "is_safe": is_safe,
                "safety_score": safety_score,
                "issues_found": issues_found,
                "action": "allow" if is_safe else "review"
            }
            
        except Exception as e:
            logger.error(f"Error in safety check: {str(e)}")
            return {
                "is_safe": False,
                "safety_score": 0,
                "issues_found": [f"Error: {str(e)}"],
                "action": "review"
            }
    
    def filter_response(self, response: str) -> str:
        """Filter and clean response if needed"""
        safety_result = self.check_content_safety(response)
        
        if not safety_result["is_safe"]:
            logger.warning(f"Unsafe content detected: {safety_result['issues_found']}")
            # Return a safe alternative response
            return "I apologize, but I cannot provide that information. Please ask about something else."
        
        return response
    
    def validate_input(self, user_input: str, input_type: str = "text") -> Dict[str, Any]:
        """Validate user input for safety and appropriateness"""
        safety_result = self.check_content_safety(user_input)
        
        validation_issues = []
        
        # Input length validation
        if len(user_input) > 10000:
            validation_issues.append("Input too long")
        elif len(user_input.strip()) == 0:
            validation_issues.append("Empty input")
        
        # URL validation (if applicable)
        if input_type == "url":
            url_pattern = r'https?://[^\s]+'
            if not re.match(url_pattern, user_input):
                validation_issues.append("Invalid URL format")
        
        is_valid = len(validation_issues) == 0 and safety_result["is_safe"]
        
        return {
            "is_valid": is_valid,
            "safety_check": safety_result,
            "validation_issues": validation_issues,
            "recommendation": "accept" if is_valid else "reject"
        }