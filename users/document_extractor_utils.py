import os
import tempfile
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import logging
from django.utils import timezone
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('document_verification')


class DjangoDocumentExtractor:
    """Document extractor integration for Django"""

    def __init__(self, config_path=None):
        """
        Initialize the extractor with optional configuration

        Args:
            config_path: Path to JSON configuration file (optional)
        """
        # Default configuration
        self.config = {
            'coordinates': {
                'x_percent': 0.55,
                'y_percent': 0.24,
                'w_percent': 0.2,
                'h_percent': 0.05
            },
            'preprocessing': {
                'alpha': 1.5,
                'beta': 10,
                'use_adaptive_threshold': True
            },
            'ocr': {
                'languages': 'eng',  # Default to English - adjust if needed
                'psm_modes': [8, 7, 6, 10],
                'use_ai_ocr': False  # Simplify by disabling AI OCR for the web app
            },
            'pattern': r'([A-Z0-9]+E)',  # Pattern to match
        }

        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    # Update default config with custom settings
                    self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

    def _update_nested_dict(self, d, u):
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def extract_from_django_file(self, file_field):
        """
        Extract identifier from a Django FileField

        Args:
            file_field: Django FileField containing PDF document

        Returns:
            The extracted identifier or None if not found
        """
        if not file_field:
            logger.warning("No file provided for extraction")
            return None

        try:
            # Create a temporary file to process
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name

                # Write the file content to the temporary file
                for chunk in file_field.chunks():
                    temp_file.write(chunk)

            # Process the temporary file
            identifier = self._process_pdf(temp_path)

            # Clean up
            os.unlink(temp_path)

            return identifier

        except Exception as e:
            logger.error(f"Error extracting from Django file: {e}")
            return None

    def _process_pdf(self, pdf_path):
        """
        Process a PDF file to extract the identifier

        Args:
            pdf_path: Path to the PDF file

        Returns:
            The extracted identifier or None if not found
        """
        try:
            # Convert PDF to image
            logger.info(f"Processing document: {os.path.basename(pdf_path)}")
            images = convert_from_path(pdf_path)
            image = np.array(images[0])

            # Get dimensions
            height, width = image.shape[:2]

            # Calculate region coordinates
            coords = self.config['coordinates']
            x = int(width * coords['x_percent'])
            y = int(height * coords['y_percent'])
            w = int(width * coords['w_percent'])
            h = int(height * coords['h_percent'])

            # Extract region of interest
            roi = image[y:y + h, x:x + w]

            # Process with OCR
            return self._process_with_tesseract(roi)

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None

    def _process_with_tesseract(self, roi):
        """Process the region with Tesseract OCR"""
        try:
            # Convert to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing
            preproc = self.config['preprocessing']
            roi_enhanced = cv2.convertScaleAbs(roi_gray, alpha=preproc['alpha'], beta=preproc['beta'])

            if preproc.get('use_adaptive_threshold', False):
                roi_enhanced = cv2.adaptiveThreshold(
                    roi_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

            # Try different PSM modes
            for psm in self.config['ocr']['psm_modes']:
                config = f'--psm {psm}'
                text = pytesseract.image_to_string(
                    roi_enhanced,
                    lang=self.config['ocr']['languages'],
                    config=config
                )
                text = text.strip()

                # Look for identifier pattern
                match = re.search(self.config['pattern'], text)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error in Tesseract processing: {e}")
            return None


def verify_association_document(association):
    """
    Verify the association's RNE document against the provided matricule_fiscal

    Args:
        association: The AssociationAccount instance to verify

    Returns:
        A tuple (is_verified, notes) indicating verification result and any notes
    """
    if not association.rne_document:
        return False, "No RNE document uploaded"

    if not association.matricule_fiscal:
        return False, "No matricule fiscal provided"

    # Create extractor
    extractor = DjangoDocumentExtractor()

    # Extract identifier from document
    extracted_id = extractor.extract_from_django_file(association.rne_document)

    if not extracted_id:
        return False, "Failed to extract identifier from document"

    # Compare with the provided matricule_fiscal
    expected_id = association.matricule_fiscal.upper()

    # Check if they match (allow for slight variations)
    if extracted_id == expected_id:
        return True, f"Document verified successfully. Extracted ID: {extracted_id}"
    else:
        # Calculate similarity for better feedback
        similarity = calculate_similarity(extracted_id, expected_id)
        if similarity > 0.8:  # 80% similarity threshold
            return True, f"Document verified with minor variations. Extracted: {extracted_id}, Expected: {expected_id}"
        else:
            return False, f"Verification failed. Extracted: {extracted_id}, Expected: {expected_id}"


def process_association_verification(association):
    """
    Perform document verification and update association verification fields

    Args:
        association: AssociationAccount instance to verify

    Returns:
        The updated AssociationAccount instance
    """
    # Perform document verification
    is_verified, notes = verify_association_document(association)

    # Update verification status based on result
    if is_verified:
        association.verification_status = 'verified'
        association.is_verified = True
        association.verification_date = datetime.now()
    else:
        association.verification_status = 'failed'
        association.is_verified = False

    # Store verification notes
    association.verification_notes = notes

    # Save changes
    association.save()

    return association
def calculate_similarity(str1, str2):
    """
    Calculate simple string similarity ratio

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    # Simple implementation - can be improved with more sophisticated algorithms
    if not str1 or not str2:
        return 0

    matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
    max_len = max(len(str1), len(str2))

    return matches / max_len if max_len > 0 else 0