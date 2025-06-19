import os
import base64
from typing import Optional, Tuple, List
from huggingface_hub import InferenceClient
from PIL import Image
import io
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pathlib import Path
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QwenOCR:
    """Qwen OCR implementation using Hugging Face Inference API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Qwen OCR
        Args:
            api_key: Fireworks AI API key. If not provided, will look for FIREWORKS_AI_API_KEY in environment
        """
        self.api_key = api_key or os.environ.get("FIREWORKS_AI_API_KEY")
        if not self.api_key:
            raise ValueError("Fireworks AI API key is required. Set FIREWORKS_AI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=self.api_key
        )
        self.model = "Qwen/Qwen2.5-VL-32B-Instruct"
        self.confidence = 0.0

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=100)  # Maximum quality
        return base64.b64encode(buffered.getvalue()).decode()

    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Increase resolution
        width, height = image.size
        new_width = width * 2
        new_height = height * 2
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Enhance contrast
        image_array = np.array(image)
        image_array = cv2.convertScaleAbs(image_array, alpha=1.2, beta=0)
        
        return Image.fromarray(image_array)

    def _clean_json_response(self, text: str) -> str:
        """Clean and extract JSON from model response"""
        try:
            # Remove any markdown code block markers
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
            
            # Find JSON object in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Validate JSON
                json.loads(json_str)  # This will raise an error if invalid
                return json_str
            
            # If no JSON found, try to construct a basic structure
            logger.warning("No valid JSON found in response, constructing basic structure")
            return json.dumps({
                "supplier_name": "",
                "invoice_number": "",
                "invoice_date": "",
                "line_items": [],
                "total_amount": 0.0
            })
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}")
            # Return basic structure on error
            return json.dumps({
                "supplier_name": "",
                "invoice_number": "",
                "invoice_date": "",
                "line_items": [],
                "total_amount": 0.0
            })

    def _process_single_image(self, image: Image.Image) -> Tuple[str, float]:
        """Process a single image with Qwen"""
        try:
            # Enhance image quality
            enhanced_image = self._enhance_image_quality(image)
            
            # Convert to base64
            image_base64 = self._image_to_base64(enhanced_image)
            
            # Create message for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this invoice image and extract information in JSON format.
                            Focus on:
                            1. Supplier name at the top of the invoice
                            2. Invoice number and date
                            3. Line items with quantities and prices
                            4. Total amount
                            
                            Format the response as a valid JSON object with this structure:
                            {
                                "supplier_name": "",
                                "invoice_number": "",
                                "invoice_date": "",
                                "line_items": [
                                    {
                                        "description": "",
                                        "quantity": 0,
                                        "unit_price": 0.0,
                                        "total_price": 0.0
                                    }
                                ],
                                "total_amount": 0.0
                            }
                            
                            Ensure the response is a valid JSON object."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # Get completion from model
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1  # Lower temperature for more consistent output
            )

            # Extract and clean response
            extracted_text = completion.choices[0].message.content
            cleaned_json = self._clean_json_response(extracted_text)
            
            # For Qwen, we'll use a default confidence score as it doesn't provide one
            self.confidence = 0.90  # Default high confidence for Qwen
            
            return cleaned_json, self.confidence

        except Exception as e:
            logger.error(f"Error in Qwen OCR: {str(e)}")
            return json.dumps({
                "supplier_name": "",
                "invoice_number": "",
                "invoice_date": "",
                "line_items": [],
                "total_amount": 0.0
            }), 0.0

    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to high-resolution images using PyMuPDF"""
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            images = []
            
            # Process each page
            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document[page_num]
                
                # Set zoom factor for higher resolution (2x)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                
                # Get page pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def extract_text(self, file_path: str) -> Tuple[str, float]:
        """
        Extract text from file using Qwen model
        Args:
            file_path: Path to the file (PDF or image)
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                # Convert PDF to high-resolution images
                images = self._convert_pdf_to_images(file_path)
                
                # Process pages in parallel
                results = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_page = {
                        executor.submit(self._process_single_image, image): i 
                        for i, image in enumerate(images)
                    }
                    
                    for future in as_completed(future_to_page):
                        try:
                            result, _ = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing PDF page: {str(e)}")
                            raise
                
                # Combine results from all pages
                combined_text = self._combine_page_results(results)
                return combined_text, self.confidence
                
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                # Load image
                image = Image.open(file_path)
                
                # Process single image
                return self._process_single_image(image)
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
        except Exception as e:
            logger.error(f"Error in extract_text: {str(e)}")
            return json.dumps({
                "supplier_name": "",
                "invoice_number": "",
                "invoice_date": "",
                "line_items": [],
                "total_amount": 0.0
            }), 0.0

    def _combine_page_results(self, results: List[str]) -> str:
        """Combine results from multiple PDF pages"""
        try:
            # Parse each result
            parsed_results = []
            for result in results:
                try:
                    # Parse JSON
                    data = json.loads(result)
                    parsed_results.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON result: {str(e)}")
                    continue
            
            if not parsed_results:
                return json.dumps({
                    "supplier_name": "",
                    "invoice_number": "",
                    "invoice_date": "",
                    "line_items": [],
                    "total_amount": 0.0
                })
            
            # Combine results
            combined = {
                "supplier_name": parsed_results[0].get("supplier_name", ""),
                "invoice_number": parsed_results[0].get("invoice_number", ""),
                "invoice_date": parsed_results[0].get("invoice_date", ""),
                "line_items": [],
                "total_amount": 0.0
            }
            
            # Combine line items and total amount
            for result in parsed_results:
                combined["line_items"].extend(result.get("line_items", []))
                combined["total_amount"] += float(result.get("total_amount", 0.0))
            
            return json.dumps(combined)
            
        except Exception as e:
            logger.error(f"Error combining page results: {str(e)}")
            return json.dumps({
                "supplier_name": "",
                "invoice_number": "",
                "invoice_date": "",
                "line_items": [],
                "total_amount": 0.0
            })

    def get_confidence(self) -> float:
        """Get confidence score for the last extraction"""
        return self.confidence

def main():
    """Test the Qwen OCR implementation"""
    # Check if API key is set
    if not os.environ.get("FIREWORKS_AI_API_KEY"):
        print("Please set FIREWORKS_AI_API_KEY environment variable")
        return

    # Initialize Qwen OCR
    qwen_ocr = QwenOCR()

    # Test with a sample document
    try:
        # Process document
        file_path = "/Users/aishwary.yadav_int/Documents/OCR task/Screenshot 2025-06-13 at 4.08.19 PM.png"
        text, confidence = qwen_ocr.extract_text(file_path)
        
        # Print results
        print("\nExtracted Text:")
        print(text)
        print(f"\nConfidence Score: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 