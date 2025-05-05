import os
import uuid
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .models import Document, DocumentChunk

# Initialize a TF-IDF vectorizer instead of sentence-transformers
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
document_vectors = None
chunk_ids = []
chunks_df = None


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return text


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split text into smaller chunks with overlap
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and text[end] != ' ':
            # Try to find a space to break on
            next_space = text.find(' ', end)
            if next_space != -1 and next_space - end < 50:  # Don't go too far
                end = next_space

        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks


def store_document_and_chunks(title, text, file_path=None, language='fr'):
    """
    Process document text, create chunks and store in the database
    """
    global document_vectors, chunk_ids, chunks_df

    # Create the document
    document = Document.objects.create(
        title=title,
        content=text,
        file=file_path,
        language=language
    )

    # Chunk the text
    chunks = chunk_text(text)

    # Store chunks
    stored_chunks = []
    for chunk in chunks:
        # Generate a unique ID for the chunk
        chunk_id = f"{document.id}_{uuid.uuid4()}"

        # Create and save the chunk
        chunk_obj = DocumentChunk.objects.create(
            document=document,
            content=chunk,
            chunk_id=chunk_id
        )
        stored_chunks.append(chunk_obj)

    return document


def setup_tfidf_index():
    """
    Load all document chunks from the database and create a TF-IDF index
    """
    global document_vectors, chunk_ids, chunks_df

    # Get all document chunks
    chunks = DocumentChunk.objects.all()

    # If no chunks exist, return None
    if not chunks.exists():
        return

    # Create a DataFrame for easier lookup
    data = []
    for chunk in chunks:
        data.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'document_id': chunk.document_id
        })

    chunks_df = pd.DataFrame(data)
    chunk_texts = chunks_df['content'].tolist()
    chunk_ids = chunks_df['chunk_id'].tolist()

    # Create TF-IDF vectors
    document_vectors = vectorizer.fit_transform(chunk_texts)


def find_relevant_chunks(query, top_k=5):
    """
    Find the most relevant document chunks for a given query using TF-IDF
    """
    global document_vectors, chunk_ids, chunks_df

    # If vectors not initialized, do it now
    if document_vectors is None or chunks_df is None:
        setup_tfidf_index()

    # If still no vectors, return empty list
    if document_vectors is None or chunks_df is None:
        return []

    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate similarity
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Get top k indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Get the corresponding chunk IDs
    relevant_chunk_ids = [chunk_ids[i] for i in top_indices]

    # Get the chunks from the database
    relevant_chunks = DocumentChunk.objects.filter(chunk_id__in=relevant_chunk_ids)

    return relevant_chunks


def process_policy_document(pdf_path):
    """
    Process the NGO policy document
    """
    # Extract text
    text = extract_text_from_pdf(pdf_path)

    # Store document and create chunks
    document = store_document_and_chunks(
        title="Décret-loi n° 2011-88 du 24 septembre 2011",
        text=text,
        file_path=pdf_path,
        language='fr'
    )

    # Initialize the TF-IDF index
    setup_tfidf_index()

    return document