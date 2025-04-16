#!/usr/bin/env python
import os
import sys
import django
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'auth.settings')
django.setup()

from chatbot.models import Document, DocumentChunk
from chatbot.utils import process_policy_document, extract_text_from_pdf
from django.conf import settings

# Check API token
api_token = os.environ.get('HUGGINGFACE_API_TOKEN', '')
if api_token:
    print(f"Hugging Face API token found: {api_token[:5]}...")
else:
    print("WARNING: No Hugging Face API token found!")

# Check if document exists and is properly processed
docs = Document.objects.filter(title__contains="Décret-loi n° 2011-88")
if docs.exists():
    doc = docs.first()
    chunks = doc.chunks.all()
    print(f"Document found: {doc.title}")
    print(f"Number of chunks: {chunks.count()}")
    
    # If no chunks, process again
    if chunks.count() == 0:
        print("No chunks found! Processing document again...")
        # Delete the document to reprocess
        doc.delete()
        docs_exist = False
    else:
        # Print sample chunk to verify content
        if chunks.exists():
            print("\nSample chunk content:")
            print(chunks.first().content[:200] + "...\n")
        docs_exist = True
else:
    docs_exist = False
    print("No document found.")

# Process document if needed
if not docs_exist:
    # Get PDF path - adjust this to your actual file path
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'policy_documents', 'decret_loi_2011_88.pdf')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: Policy document not found at {pdf_path}")
        print("Please save the document to this location and run the script again.")
        sys.exit(1)
    
    print(f"Processing policy document from {pdf_path}...")
    
    # Extract and print a sample of the text to verify it's working
    sample_text = extract_text_from_pdf(pdf_path)
    print(f"Sample text from PDF: {sample_text[:200]}...")
    
    try:
        # Process the document
        document = process_policy_document(pdf_path)
        print(f"Successfully processed document: {document.title}")
        print(f"Created {document.chunks.count()} document chunks")
    except Exception as e:
        print(f"Error processing document: {e}")

# Test retrieval
from chatbot.utils import find_relevant_chunks
test_query = "Comment créer une association en Tunisie?"
chunks = find_relevant_chunks(test_query, top_k=3)
print(f"\nTest query: {test_query}")
print(f"Found {len(chunks)} relevant chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.content[:150] + "...")

print("\nInitialization complete!")