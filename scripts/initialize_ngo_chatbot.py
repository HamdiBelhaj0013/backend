#!/usr/bin/env python
"""
Script to initialize the NGO chatbot with the policy document.
Run this script after setting up the Django app and migrating the database.

Usage:
    python scripts/initialize_ngo_chatbot.py
"""

import os
import sys
import django
from django.conf import settings

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auth.settings')
django.setup()

from chatbot.models import Document
from chatbot.utils import process_policy_document


def initialize_chatbot():
    """Initialize the chatbot with the NGO policy document"""

    # Check if document is already processed
    if Document.objects.filter(title__contains="Décret-loi n° 2011-88").exists():
        print("Policy document already exists in the database.")
        return

    # Get the path to the policy document
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'policy_documents', 'decret_loi_2011_88.pdf')

    # Ensure directory exists
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Policy document not found at {pdf_path}")
        print("Please save the document to this location and run the script again.")
        return

    print(f"Processing policy document from {pdf_path}...")

    try:
        # Process the document
        document = process_policy_document(pdf_path)
        print(f"Successfully processed document: {document.title}")
        print(f"Created {document.chunks.count()} document chunks with embeddings")
    except Exception as e:
        print(f"Error processing document: {e}")


if __name__ == "__main__":
    initialize_chatbot()