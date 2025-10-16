import logging
import os
from typing import Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class LLMManager:
    """Gemini-based LLM Manager for text generation and embeddings"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'gemini-2.5-flash')
        self.api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            logger.info(f"Initializing Gemini model: {self.model_name}")
            
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings
            )
            
            logger.info("Gemini model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
    
    def set_generation_config(self, **kwargs):
        """Set generation configuration parameters"""
        self.config.update(kwargs)
        logger.info(f"Updated generation config: {kwargs}")
    
    def generate(self, 
                prompt: str, 
                max_output_tokens: int = None,
                temperature: float = None,
                **kwargs) -> str:
        """Generate text using Gemini"""
        try:
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens or self.config.get('max_output_tokens', 1024),
                temperature=temperature or self.config.get('temperature', 0.7),
                top_p=kwargs.get('top_p', self.config.get('top_p', 0.9)),
                top_k=kwargs.get('top_k', self.config.get('top_k', 40))
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "No response generated"
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str) -> list:
        """Get text embedding using Gemini embedding model"""
        try:
            # Use Gemini's embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """Get current LLM configuration"""
        return {
            "model_name": self.model_name,
            "max_output_tokens": self.config.get('max_output_tokens', 1024),
            "temperature": self.config.get('temperature', 0.7),
            "top_p": self.config.get('top_p', 0.9),
            "top_k": self.config.get('top_k', 40)
        }
    
    def get_crewai_llm(self):
        """Get LLM instance for CrewAI integration"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.config.get('temperature', 0.7),
                max_output_tokens=self.config.get('max_output_tokens', 1024)
            )
            
        except Exception as e:
            logger.error(f"Error creating CrewAI LLM: {str(e)}")
            raise