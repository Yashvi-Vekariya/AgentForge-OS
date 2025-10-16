import logging
import os
from PIL import Image
import google.generativeai as genai
from typing import Optional, Dict, Any
import io

logger = logging.getLogger(__name__)

class GeminiVisionHelper:
    """Gemini Vision helper for image understanding and captioning"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini Vision model"""
        try:
            logger.info(f"Initializing Gemini Vision model: {self.model_name}")
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini Vision model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini Vision model: {str(e)}")
            raise
    
    def generate_image_caption(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate caption for an image using Gemini Vision"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Prepare prompt
            if prompt:
                text_prompt = prompt
            else:
                text_prompt = "Describe this image in detail. Provide a comprehensive caption that captures the main elements, objects, people, actions, and setting visible in the image."
            
            # Generate caption using Gemini Vision
            response = self.model.generate_content([text_prompt, image])
            
            if response.candidates and response.candidates[0].content.parts:
                caption = response.candidates[0].content.parts[0].text.strip()
            else:
                caption = "No caption generated"
            
            return {
                "caption": caption,
                "model": self.model_name,
                "prompt_used": prompt,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating image caption: {str(e)}")
            return {
                "caption": "",
                "error": str(e),
                "status": "failed"
            }
    
    def visual_question_answering(self, image_path: str, question: str) -> Dict[str, Any]:
        """Answer questions about an image using Gemini Vision"""
        try:
            image = Image.open(image_path)
            
            # Format prompt for VQA
            prompt = f"Please answer this question about the image: {question}"
            
            # Generate answer using Gemini Vision
            response = self.model.generate_content([prompt, image])
            
            if response.candidates and response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text.strip()
            else:
                answer = "No answer generated"
            
            return {
                "question": question,
                "answer": answer,
                "model": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in visual question answering: {str(e)}")
            return {
                "question": question,
                "answer": "",
                "error": str(e),
                "status": "failed"
            }
    
    def describe_image_detailed(self, image_path: str) -> Dict[str, Any]:
        """Generate detailed description of an image using Gemini Vision"""
        try:
            image = Image.open(image_path)
            
            prompt = """Please provide a comprehensive and detailed description of this image. Include:
            1. Main objects, people, and subjects in the image
            2. Actions or activities taking place
            3. Setting, location, and environment
            4. Colors, lighting, and visual composition
            5. Style, mood, and atmosphere
            6. Any text visible in the image
            7. Spatial relationships between elements
            
            Provide a thorough analysis that would help someone who cannot see the image understand it completely."""
            
            response = self.model.generate_content([prompt, image])
            
            if response.candidates and response.candidates[0].content.parts:
                detailed_description = response.candidates[0].content.parts[0].text.strip()
            else:
                detailed_description = "No detailed description generated"
            
            return {
                "detailed_description": detailed_description,
                "model": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed description: {str(e)}")
            return {
                "detailed_description": "",
                "error": str(e),
                "status": "failed"
            }

# For backward compatibility
BLIP2Helper = GeminiVisionHelper