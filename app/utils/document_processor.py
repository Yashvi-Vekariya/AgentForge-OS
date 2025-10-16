"""
Document processing utilities for various file formats
"""

import logging
import chardet
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats and extract text content"""
    
    @staticmethod
    def detect_encoding(content: bytes) -> str:
        """Detect the encoding of text content"""
        try:
            result = chardet.detect(content)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                encoding = 'utf-8'
                
            return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    @staticmethod
    def extract_text_from_bytes(content: bytes, filename: str = "") -> Tuple[str, str]:
        """
        Extract text from bytes content based on file type
        Returns: (text_content, file_type)
        """
        filename_lower = filename.lower() if filename else ""
        
        try:
            if filename_lower.endswith('.pdf'):
                return DocumentProcessor._extract_from_pdf(content), 'pdf'
            elif filename_lower.endswith(('.doc', '.docx')):
                return DocumentProcessor._extract_from_word(content), 'word'
            elif filename_lower.endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml')):
                return DocumentProcessor._extract_from_text(content), 'text'
            else:
                # Try to process as text
                text_content = DocumentProcessor._extract_from_text(content)
                return text_content, 'text'
                
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            raise
    
    @staticmethod
    def _extract_from_pdf(content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            import PyPDF2
            import io
            
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                return f"PDF file processed but no text content could be extracted. The PDF might contain only images or be password protected."
            
            return text_content.strip()
            
        except ImportError:
            return f"PDF processing not available. Please install PyPDF2: pip install PyPDF2"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return f"Error processing PDF: {str(e)}"
    
    @staticmethod
    def _extract_from_word(content: bytes) -> str:
        """Extract text from Word document content"""
        try:
            import docx
            import io
            
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)
            
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            if not text_content.strip():
                return f"Word document processed but no text content found."
            
            return text_content.strip()
            
        except ImportError:
            return f"Word document processing not available. Please install python-docx: pip install python-docx"
        except Exception as e:
            logger.error(f"Error extracting Word document text: {e}")
            return f"Error processing Word document: {str(e)}"
    
    @staticmethod
    def _extract_from_text(content: bytes) -> str:
        """Extract text from plain text content with encoding detection"""
        # Try UTF-8 first
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            pass
        
        # Detect encoding
        encoding = DocumentProcessor.detect_encoding(content)
        
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            # Last resort: decode with errors='ignore'
            logger.warning("Using fallback decoding with errors='ignore'")
            return content.decode('utf-8', errors='ignore')
    
    @staticmethod
    def get_supported_formats() -> list:
        """Get list of supported file formats"""
        return [
            '.txt', '.md', '.pdf', '.doc', '.docx',
            '.py', '.js', '.html', '.css', '.json', '.xml'
        ]
    
    @staticmethod
    def is_supported_format(filename: str) -> bool:
        """Check if file format is supported"""
        if not filename:
            return False
        
        filename_lower = filename.lower()
        supported_formats = DocumentProcessor.get_supported_formats()
        
        return any(filename_lower.endswith(fmt) for fmt in supported_formats)
