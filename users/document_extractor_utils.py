import os
import tempfile
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import logging
from django.utils import timezone
import cv2
import numpy as np
import re
import json
from datetime import datetime
import time

# Import the optimized document extractor from ocr2
from pdf2image import convert_from_path
import pytesseract


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('document_verification')


class OptimizedDocumentExtractor:
    """Optimized document extractor with regions specially tuned for the document type"""

    def __init__(self, debug_mode=False):
        """Initialize the optimized extractor with the best-performing settings"""
        self.debug_mode = debug_mode


        self.config = {
            # Optimal regions based on comprehensive analysis
            'regions': [

                {
                    'x_percent': 0.53,
                    'y_percent': 0.23,
                    'w_percent': 0.22,
                    'h_percent': 0.055
                },
                # Secondary region - slightly wider
                {
                    'x_percent': 0.5,
                    'y_percent': 0.21,
                    'w_percent': 0.28,
                    'h_percent': 0.07
                },
                # Tertiary region - covers more vertical space
                {
                    'x_percent': 0.52,
                    'y_percent': 0.19,
                    'w_percent': 0.24,
                    'h_percent': 0.09
                },
            ],

            'preprocessing': [
                # Standard preprocessing - works for most documents
                {
                    'name': 'standard',
                    'alpha': 1.5,  # Contrast enhancement
                    'beta': 10,  # Brightness adjustment
                    'use_adaptive_threshold': True,
                    'threshold_block_size': 11,
                    'threshold_c': 2
                },
                # High contrast version - for faded or low contrast documents
                {
                    'name': 'high_contrast',
                    'alpha': 2.0,
                    'beta': 0,
                    'use_adaptive_threshold': True,
                    'threshold_block_size': 15,
                    'threshold_c': 5
                },
                # Binary version - for maximum contrast
                {
                    'name': 'binary',
                    'use_otsu': True
                }
            ],
            # OCR settings
            'ocr': {
                'languages': 'ara+eng',
                'psm_modes': [7, 8, 6, 4],
                'dpi': 300
            },

            'patterns': [
                r'(\d{7}[A-Z])',
                r'(\d{7}[A-Za-z])',
                r'[N]?(\d{7}[A-Z])',
                r'(\d{7}[|l])',
                r'N?(\d{7}[A-Za-z])'
            ]
        }

    def extract_from_document(self, pdf_path, output_dir=None):

        try:
            start_time = time.time()

            # Create debug directory if needed
            if self.debug_mode and output_dir:
                debug_dir = os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf', ''))
                os.makedirs(debug_dir, exist_ok=True)
            else:
                debug_dir = None

            # Convert PDF to image
            logger.debug(f"Processing: {os.path.basename(pdf_path)}")
            images = convert_from_path(pdf_path, dpi=self.config['ocr']['dpi'])
            image = np.array(images[0])


            height, width = image.shape[:2]


            all_results = []
            confidence_map = {}


            for region_idx, region in enumerate(self.config['regions']):

                x = int(width * region['x_percent'])
                y = int(height * region['y_percent'])
                w = int(width * region['w_percent'])
                h = int(height * region['h_percent'])


                roi = image[y:y + h, x:x + w]


                if debug_dir:
                    self._save_region_visualization(image, x, y, w, h, debug_dir, f"region_{region_idx}")

                # Process with different preprocessing methods
                for preproc_idx, preproc in enumerate(self.config['preprocessing']):
                    processed_roi = self._preprocess_image(roi, preproc)

                    # Save processed image if in debug mode
                    if debug_dir:
                        cv2.imwrite(
                            os.path.join(debug_dir, f"region_{region_idx}_preproc_{preproc['name']}.jpg"),
                            processed_roi
                        )


                    for psm_mode in self.config['ocr']['psm_modes']:
                        if region_idx > 1 and psm_mode not in [7, 6]:
                            continue

                        region_results = self._process_with_ocr(processed_roi, psm_mode, debug_dir,
                                                                f"region_{region_idx}_preproc_{preproc['name']}_psm_{psm_mode}")


                        for result in region_results:
                            all_results.append(result)
                            confidence_map[result] = confidence_map.get(result, 0) + 1

                        if region_results and region_idx == 0 and preproc_idx == 0 and psm_mode == 7:
                            if len(region_results) == 1:
                                identifier = region_results[0]
                                confidence = 100.0
                                elapsed_time = time.time() - start_time

                                return {
                                    'identifier': identifier,
                                    'confidence': confidence,
                                    'processing_time': elapsed_time,
                                    'filename': os.path.basename(pdf_path)
                                }


            identifier = None
            confidence = 0

            if all_results:
                # Find the most common result
                result_counts = {}
                for result in all_results:
                    result_counts[result] = result_counts.get(result, 0) + 1


                max_count = 0
                for result, count in result_counts.items():
                    if count > max_count:
                        max_count = count
                        identifier = result


                confidence = (max_count / len(all_results)) * 100


            elapsed_time = time.time() - start_time

            return {
                'identifier': identifier,
                'confidence': confidence,
                'processing_time': elapsed_time,
                'filename': os.path.basename(pdf_path)
            }

        except Exception as e:
            logger.error(f"Error extracting from document: {e}")
            return {
                'identifier': None,
                'confidence': 0.0,
                'processing_time': 0.0,
                'filename': os.path.basename(pdf_path)
            }

    def _preprocess_image(self, image, preproc):
        """Apply preprocessing to the image"""
        try:
            # Ensure we have a valid image
            if image is None or image.size == 0:
                logger.warning("Invalid image for preprocessing")
                return np.zeros((10, 10), dtype=np.uint8)

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply preprocessing based on the method
            if preproc.get('use_otsu', False):
                # Otsu thresholding
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return processed

            # Apply contrast/brightness adjustment
            if 'alpha' in preproc:
                processed = cv2.convertScaleAbs(
                    gray,
                    alpha=preproc.get('alpha', 1.0),
                    beta=preproc.get('beta', 0)
                )
            else:
                processed = gray.copy()

            # Apply adaptive threshold if configured
            if preproc.get('use_adaptive_threshold', False):
                block_size = preproc.get('threshold_block_size', 11)
                c = preproc.get('threshold_c', 2)

                return cv2.adaptiveThreshold(
                    processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c
                )

            return processed

        except Exception as e:
            logger.warning(f"Error in preprocessing: {e}")
            return image  # Return original image on error

    def _process_with_ocr(self, image, psm_mode, debug_dir=None, name=None):
        """Process image with OCR and extract IDs"""
        results = []

        try:
            # Skip processing if image is invalid
            if image is None or image.size == 0:
                return results

            # Build OCR configuration
            config = f'--psm {psm_mode} --dpi {self.config["ocr"]["dpi"]}'

            # Run OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.config['ocr']['languages'],
                config=config
            )

            # Log OCR text if in debug mode
            if debug_dir and name:
                with open(os.path.join(debug_dir, "ocr_log.txt"), "a") as f:
                    f.write(f"OCR for {name}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(text)
                    f.write("\n" + "=" * 40 + "\n\n")

            # Extract IDs using patterns
            for pattern in self.config['patterns']:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Clean up and validate match
                    cleaned_match = self._clean_and_validate_id(match)
                    if cleaned_match:
                        results.append(cleaned_match)

            return results

        except Exception as e:
            logger.warning(f"Error in OCR processing: {e}")
            return results

    def _clean_and_validate_id(self, id_text):
        """Clean and validate an ID"""
        try:
            # Handle IDs with potential OCR issues
            if len(id_text) == 8:
                # Ensure first 7 characters are digits
                for i in range(7):
                    if not id_text[i].isdigit():
                        # Try to fix common OCR errors (e.g., 'l' for '1')
                        if id_text[i] in 'lI|':
                            id_text = id_text[:i] + '1' + id_text[i + 1:]
                        else:
                            return None  # Invalid ID format

                # Ensure last character is a letter
                if not id_text[7].isalpha():
                    return None  # Invalid ID format

                # Convert last character to uppercase
                id_text = id_text[:7] + id_text[7].upper()

                return id_text

            return None  # Not the right length

        except Exception as e:
            logger.debug(f"Error validating ID: {e}")
            return None

    def _save_region_visualization(self, image, x, y, w, h, output_dir, name):
        """Save visualization of the extraction region"""
        try:
            # Create a copy of the image to draw on
            vis_img = image.copy()

            # Draw rectangle for the region of interest
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the visualization
            output_path = os.path.join(output_dir, f"{name}_region.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

            # Also save just the region
            region_img = image[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(output_dir, f"{name}_extracted.jpg"),
                        cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logger.warning(f"Error saving region visualization: {e}")


class DjangoOptimizedExtractor:
    """Django integration for the optimized document extractor"""

    def __init__(self, debug_mode=False):
        """Initialize with Django integration support"""
        self.extractor = OptimizedDocumentExtractor(debug_mode=debug_mode)
        self.debug_mode = debug_mode

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
            return None, 0.0

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name

                # Write the file content
                for chunk in file_field.chunks():
                    temp_file.write(chunk)

            # Create debug directory if needed
            debug_dir = None
            if self.debug_mode:
                debug_dir = tempfile.mkdtemp(prefix='django_ocr_debug_')

            # Extract data from the document
            result = self.extractor.extract_from_document(temp_path, debug_dir)

            # Clean up
            os.unlink(temp_path)

            return result.get('identifier'), result.get('confidence', 0.0)

        except Exception as e:
            logger.error(f"Error extracting from Django file: {e}")
            return None, 0.0


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

    # Create extractor with debug mode off
    extractor = DjangoOptimizedExtractor(debug_mode=False)

    # Extract identifier from document
    extracted_id, confidence = extractor.extract_from_django_file(association.rne_document)

    if not extracted_id:
        return False, "Failed to extract identifier from document"

    # Compare with the provided matricule_fiscal
    expected_id = association.matricule_fiscal.upper()

    # Check if they match (using both exact match and similarity)
    if extracted_id == expected_id:
        return True, f"Document verified successfully. Extracted ID: {extracted_id} (Confidence: {confidence:.1f}%)"
    else:
        # Calculate similarity for better feedback
        similarity = calculate_similarity(extracted_id, expected_id)

        if similarity > 0.8:  # 80% similarity threshold
            return True, f"Document verified with minor variations. Extracted: {extracted_id}, Expected: {expected_id} (Similarity: {similarity:.1f}, Confidence: {confidence:.1f}%)"
        elif similarity > 0.6:  # 60% threshold for partial matches
            return False, f"Partial match detected. Extracted: {extracted_id}, Expected: {expected_id} (Similarity: {similarity:.1f}, Confidence: {confidence:.1f}%)"
        else:
            return False, f"Verification failed. Extracted: {extracted_id}, Expected: {expected_id} (Similarity: {similarity:.1f}, Confidence: {confidence:.1f}%)"


def process_association_verification(association):
    """
    Perform document verification and update association verification fields

    Args:
        association: AssociationAccount instance to verify

    Returns:
        The updated AssociationAccount instance
    """
    # Record start time for performance tracking
    start_time = time.time()

    # Perform document verification
    is_verified, notes = verify_association_document(association)

    # Update verification status based on result
    if is_verified:
        association.verification_status = 'verified'
        association.is_verified = True
        # Use Django's timezone-aware datetime
        association.verification_date = timezone.now()
    else:
        association.verification_status = 'failed'
        association.is_verified = False

    # Store verification notes
    association.verification_notes = notes

    # Add processing time to notes
    processing_time = time.time() - start_time
    association.verification_notes += f"\nProcessing time: {processing_time:.2f} seconds"

    # Save changes
    association.save()

    # Log the result
    logger.info(f"Association verification result for {association.name}: {association.verification_status}")
    logger.info(f"Verification notes: {association.verification_notes}")

    return association


def calculate_similarity(str1, str2):
    """
    Calculate string similarity ratio with improved handling of common OCR errors

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    if not str1 or not str2:
        return 0

    # Normalize strings to handle common OCR errors
    def normalize(s):
        replacements = {
            'l': '1', 'I': '1', '|': '1',
            'O': '0', 'o': '0',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'G': '6', 'g': '6',
            'B': '8', 'b': '8'
        }

        result = ''
        for c in s:
            result += replacements.get(c, c)

        return result.upper()  # Convert to uppercase for case-insensitive comparison


    norm1 = normalize(str1)
    norm2 = normalize(str2)


    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Calculate cost - 0 if characters match, 1 otherwise
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    max_len = max(len(norm1), len(norm2))
    if max_len == 0:
        return 1.0  # Both strings are empty

    distance = levenshtein(norm1, norm2)
    similarity = 1.0 - (distance / max_len)

    return similarity