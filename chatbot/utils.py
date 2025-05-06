import os
import uuid
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from .models import Document, DocumentChunk

# Initialize a TF-IDF vectorizer with French stopwords and better tokenization for legal text
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=['le', 'la', 'les', 'un', 'une', 'des', 'et', 'en', 'du', 'de', 'à', 'au', 'aux'],
    ngram_range=(1, 2),  # Include bigrams for legal phrases
    min_df=2,  # Minimum document frequency
    max_df=0.85  # Maximum document frequency
)
document_vectors = None
chunk_ids = []
chunks_df = None


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file with improved formatting preservation
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # Extract text with better layout preservation
            text += page.get_text("text")  # Use text mode for better layout
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return text


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Split text into smaller chunks with enhanced article awareness for legal documents
    """
    # Try to identify article boundaries for legal documents
    import re

    # More robust pattern for French legal texts
    article_pattern = r'Art\.\s+\d+\s+-|Article\s+\d+\s*[-:]'
    articles = re.split(article_pattern, text)

    # If we found clear article boundaries and there are at least a few articles
    if len(articles) > 2:  # First element is usually text before first article
        chunks = []

        # Process each article
        for i in range(1, len(articles)):
            # Extract article number from the original text
            match = re.search(r'Art\.\s+(\d+)|Article\s+(\d+)', text)
            article_num = match.group(1) if match and match.group(1) else str(i)

            article_text = f"Art. {article_num} - {articles[i]}"

            # If article is very long, further chunk it while preserving context
            if len(article_text) > chunk_size * 1.5:
                sub_chunks = []
                start = 0
                while start < len(article_text):
                    end = min(start + chunk_size, len(article_text))
                    if end < len(article_text) and article_text[end] != ' ':
                        # Try to find a sentence boundary to break on
                        sentence_end = max(
                            article_text.rfind('. ', start, end),
                            article_text.rfind('? ', start, end),
                            article_text.rfind('! ', start, end),
                            article_text.rfind('.\n', start, end)
                        )
                        if sentence_end != -1 and sentence_end > start + 100:  # Ensure minimum chunk size
                            end = sentence_end + 1

                    # Always include article number in each chunk for context
                    if start > 0:
                        article_prefix = f"Art. {article_num}"
                        sub_chunks.append(f"{article_prefix} (suite) {article_text[start:end]}")
                    else:
                        sub_chunks.append(article_text[start:end])

                    start = end - chunk_overlap

                chunks.extend(sub_chunks)
            else:
                chunks.append(article_text)

        return chunks

    # Fall back to the original method if no clear article structure is found
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and text[end] != ' ':
            # Try to find a sentence boundary to break on
            sentence_end = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end),
                text.rfind('.\n', start, end)
            )
            if sentence_end != -1 and sentence_end > start + 100:  # Ensure minimum chunk size
                end = sentence_end + 1

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

    # Rebuild the TF-IDF index after adding new document
    setup_tfidf_index()

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


def setup_semantic_model():
    """
    Initialize a sentence transformer model for semantic search
    """
    try:
        from sentence_transformers import SentenceTransformer
        # Use a multilingual model optimized for French and legal text
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return model
    except ImportError:
        print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
        return None


# Initialize semantic model
semantic_model = setup_semantic_model()
semantic_embeddings = None


def create_semantic_embeddings():
    """
    Create semantic embeddings for all document chunks
    """
    global semantic_model, semantic_embeddings, chunks_df

    if semantic_model is None:
        return None

    # Get all document chunks
    chunks = DocumentChunk.objects.all()

    # If no chunks exist, return None
    if not chunks.exists():
        return None

    # Get text of all chunks
    chunk_texts = [chunk.content for chunk in chunks]

    # Create embeddings
    embeddings = semantic_model.encode(chunk_texts, show_progress_bar=True)

    # Store embeddings in global variable
    semantic_embeddings = embeddings

    # Also store in database if embedding field exists
    for i, chunk in enumerate(chunks):
        # Convert numpy array to list for JSON storage
        chunk.embedding = embeddings[i].tolist()
        chunk.save()

    return embeddings


def expand_query_with_terms(query):
    """
    Expand query with legal terminology to improve search relevance
    """
    # Map common terms to legal terminology
    legal_term_map = {
        "créer": ["constitution", "création", "former"],
        "statuts": ["règlement", "statut", "organisation"],
        "financement": ["ressources", "financer", "budget", "cotisations"],
        "dissolution": ["liquider", "dissoudre", "disparition"],
        "membres": ["adhérents", "adhésion", "participation"],
        "administration": ["bureau", "conseil", "gestion"],
        "assemblée": ["réunion", "ag", "assemblée générale"]
    }

    expanded_terms = []
    query_lower = query.lower()

    # Add related terms if the original term appears in the query
    for term, expansions in legal_term_map.items():
        if term in query_lower:
            expanded_terms.extend(expansions)

    # Combine original query with expanded terms
    if expanded_terms:
        expanded_query = f"{query} {' '.join(expanded_terms)}"
        return expanded_query

    return query


def find_relevant_chunks(query, top_k=5):
    """
    Find the most relevant document chunks for a given query using hybrid search
    combining TF-IDF and semantic similarity with higher weight on document content
    """
    global document_vectors, chunk_ids, chunks_df, semantic_model, semantic_embeddings

    # If vectors not initialized, do it now
    if document_vectors is None or chunks_df is None:
        setup_tfidf_index()

    # If still no vectors, return empty list
    if document_vectors is None or chunks_df is None:
        return []

    # Expand query with legal terminology
    expanded_query = expand_query_with_terms(query)

    # Initialize results containers
    tfidf_scores = np.zeros(len(chunk_ids))
    semantic_scores = np.zeros(len(chunk_ids))

    # TF-IDF scoring
    query_vector = vectorizer.transform([expanded_query])
    tfidf_scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Semantic scoring (if available)
    if semantic_model is not None:
        # Initialize semantic embeddings if not done already
        if semantic_embeddings is None:
            semantic_embeddings = create_semantic_embeddings()

        if semantic_embeddings is not None:
            query_embedding = semantic_model.encode([query])[0]

            # Calculate cosine similarity for semantic embeddings
            for i, chunk_embedding in enumerate(semantic_embeddings):
                # Normalize vectors for cosine similarity
                norm_query = np.linalg.norm(query_embedding)
                norm_chunk = np.linalg.norm(chunk_embedding)

                if norm_query > 0 and norm_chunk > 0:
                    semantic_scores[i] = np.dot(query_embedding, chunk_embedding) / (norm_query * norm_chunk)

    # Combine scores with higher weight (0.7) to TF-IDF (keyword matching) to prioritize document content
    if semantic_embeddings is not None:
        combined_scores = 0.7 * tfidf_scores + 0.3 * semantic_scores
    else:
        combined_scores = tfidf_scores

    # Boost scores for chunks with article references
    import re
    for i, chunk_id in enumerate(chunk_ids):
        chunk_content = chunks_df.loc[chunks_df['chunk_id'] == chunk_id, 'content'].iloc[0]
        # Check if chunk contains article reference
        if re.search(r'Art\.\s+\d+|Article\s+\d+', chunk_content):
            combined_scores[i] *= 1.2  # Boost by 20%

    # Get top k indices
    top_indices = combined_scores.argsort()[-top_k:][::-1]

    # Get the corresponding chunk IDs
    relevant_chunk_ids = [chunk_ids[i] for i in top_indices]

    # Get the chunks from the database
    relevant_chunks = DocumentChunk.objects.filter(chunk_id__in=relevant_chunk_ids)

    # Store the scores for each chunk to sort them later
    chunks_with_scores = []
    for chunk in relevant_chunks:
        idx = chunk_ids.index(chunk.chunk_id)
        chunks_with_scores.append((chunk, combined_scores[idx]))

    sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)

    # Return only the chunks, not the scores
    return [chunk for chunk, score in sorted_chunks]


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

    # Generate semantic embeddings
    create_semantic_embeddings()

    return document