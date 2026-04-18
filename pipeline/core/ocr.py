"""OCR wrapper using Tesseract."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Extract text from images using Tesseract."""

    def __init__(self, languages: str = None):
        """
        Initialize OCR processor.
        
        Args:
            languages: Tesseract language codes (e.g., "eng+hin+tel")
        """
        self.languages = languages or os.getenv("OCR_IMAGE_LANGS", "eng")
        self.initialized = False
    
    def _initialize(self):
        """Lazy initialize pytesseract."""
        if self.initialized:
            return
        
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
            self.initialized = True
            logger.info(f"OCR initialized with languages: {self.languages}")
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            raise
    
    def extract(self, image_path: str, languages: Optional[str] = None) -> str:
        """
        Extract text from image.
        
        Args:
            image_path: Path to image file
            languages: Optional override for language codes
            
        Returns:
            Extracted text
        """
        self._initialize()
        
        lang = languages or self.languages
        
        try:
            img = self.Image.open(image_path)
            text = self.pytesseract.image_to_string(img, lang=lang)
            text = text.strip()
            
            if not text:
                logger.warning("OCR returned empty text")
            
            logger.info(f"OCR extracted {len(text)} characters from {image_path}")
            return text
        
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
