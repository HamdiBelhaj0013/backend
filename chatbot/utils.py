import os
import uuid
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import re
import string
from difflib import SequenceMatcher
from .models import Document, DocumentChunk

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize a TF-IDF vectorizer with improved settings
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='french',
    ngram_range=(1, 3),  # Include trigrams for better context
    analyzer='char_wb',  # Character n-grams including word boundaries
    min_df=2,  # Minimum document frequency
    max_df=0.85  # Ignore terms that appear in >85% of documents
)
document_vectors = None
chunk_ids = []
chunks_df = None

# Flags for advanced features
use_sentence_transformers = False
spell_checker_available = False
use_fasttext = False

# Try to import advanced libraries
try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    use_sentence_transformers = True
    logger.info("SentenceTransformer loaded successfully")
except ImportError:
    logger.warning("SentenceTransformer not available, falling back to TF-IDF")

try:
    from spellchecker import SpellChecker

    spell = SpellChecker(language='fr')  # French for Tunisian context
    spell_checker_available = True
    logger.info("SpellChecker loaded successfully")
except ImportError:
    logger.warning("SpellChecker not available, spell correction disabled")

try:
    import fasttext

    # Load French fastText model if available (need to download first)
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'cc.fr.300.bin')
    if os.path.exists(model_path):
        fasttext_model = fasttext.load_model(model_path)
        use_fasttext = True
        logger.info("FastText model loaded successfully for semantic matching")
    else:
        logger.warning("FastText model file not found")
except ImportError:
    logger.warning("FastText not available, semantic matching disabled")

# Common typos and corrections in French
COMMON_CORRECTIONS = {
    'assosiations': 'associations',
    'asociation': 'association',
    'assoc': 'association',
    'financemnt': 'financement',
    'finan': 'finance',
    'creér': 'créer',
    'crer': 'créer',
    'adherer': 'adhérer',
    'adhésion': 'adhésion',
    'disolusion': 'dissolution',
    'disolution': 'dissolution',
    'disoudre': 'dissoudre',
    'tunisie': 'tunisie',
    'statute': 'statut',
    'statutoire': 'statutaire',
    'loi': 'loi',
    'décret': 'décret',
    'juridic': 'juridique',
}

# Domain-specific vocabulary for associations in Tunisia
DOMAIN_VOCAB = [
    'association', 'statut', 'adhésion', 'adhérent', 'membre', 'cotisation',
    'dissolution', 'financement', 'subvention', 'décret-loi', 'créer', 'fondateur',
    'tunisie', 'tunisien', 'président', 'secrétaire', 'trésorier', 'assemblée',
    'générale', 'conseil', 'administration', 'vote', 'quorum', 'siège', 'social',
    'déclaration', 'publication', 'JORT', 'tribunal', 'compte', 'bancaire',
    'rapport', 'moral', 'financier', 'bilan', 'projet', 'activité', 'organisation'
]


def similar(a, b):
    """Calculate string similarity ratio using SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()


def preprocess_query(query):
    """
    Enhanced preprocessing of user query to handle typos, grammar errors and variations
    With semantic matching and context awareness
    """
    if not query:
        return query

    # Store original for logging
    original_query = query

    try:
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Fix common typos using the dictionary
        words = query.split()
        for i, word in enumerate(words):
            # Check direct common typo corrections
            if word in COMMON_CORRECTIONS:
                words[i] = COMMON_CORRECTIONS[word]
                continue

            # Try to find similar domain-specific words
            best_match = None
            best_score = 0.8  # Threshold for similarity

            for vocab_word in DOMAIN_VOCAB:
                score = similar(word, vocab_word)
                if score > best_score:
                    best_match = vocab_word
                    best_score = score

            if best_match:
                words[i] = best_match

        # Reconstruct query after typo correction
        corrected_query = ' '.join(words)

        # Further spelling correction with pyspellchecker
        if spell_checker_available:
            words = corrected_query.split()
            spell_corrected = []

            for word in words:
                # Only correct if word is misspelled and not a number or special term
                if spell.unknown([word]) and not word.isdigit() and len(word) > 2:
                    correction = spell.correction(word)
                    spell_corrected.append(correction)
                else:
                    spell_corrected.append(word)

            corrected_query = ' '.join(spell_corrected)

        # Expand common abbreviations
        abbreviations = {
            'assoc': 'association',
            'asso': 'association',
            'org': 'organisation',
            'admin': 'administration',
            'docs': 'documents',
            'doc': 'document',
            'fin': 'finance',
            'pdt': 'président',
            'sg': 'secrétaire général',
            'ag': 'assemblée générale',
            'ca': 'conseil administration',
        }

        for abbr, full in abbreviations.items():
            # Use word boundary regex to ensure we're not replacing inside words
            pattern = r'\b{}\b'.format(abbr)
            corrected_query = re.sub(pattern, full, corrected_query)

        # Log if any corrections were made
        if corrected_query != original_query.lower():
            logger.info(f"Corrected query: '{original_query}' -> '{corrected_query}'")

        return corrected_query

    except Exception as e:
        logger.error(f"Error in enhanced query preprocessing: {e}")
        return query  # Return original query if any error occurs


def extract_intent_keywords(query):
    """Extract key intent words from query for better matching"""
    # Common intent keywords in French for associations
    creation_words = ['créer', 'fonder', 'établir', 'constituer', 'former', 'lancer']
    finance_words = ['financer', 'budget', 'argent', 'subvention', 'don', 'cotisation']
    legal_words = ['loi', 'légal', 'juridique', 'règlement', 'décret', 'obligation']
    dissolution_words = ['dissoudre', 'terminer', 'fermer', 'clôturer', 'arrêter']

    keywords = []
    query_words = query.lower().split()

    # Extract matching intent words
    for word in query_words:
        if any(similar(word, intent_word) > 0.8 for intent_word in creation_words):
            keywords.append('création')
        if any(similar(word, intent_word) > 0.8 for intent_word in finance_words):
            keywords.append('finance')
        if any(similar(word, intent_word) > 0.8 for intent_word in legal_words):
            keywords.append('légal')
        if any(similar(word, intent_word) > 0.8 for intent_word in dissolution_words):
            keywords.append('dissolution')

    return list(set(keywords))  # Remove duplicates


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file with improved error handling and OCR fallback
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)

        # Extract text with metadata
        metadata = doc.metadata
        logger.info(f"Processing PDF: {os.path.basename(pdf_path)}, Pages: {len(doc)}")

        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            # If page text is very short, it might be an image-only page
            if len(page_text.strip()) < 50 and page.get_images():
                logger.info(f"Page {page_num} may contain images with little text, consider OCR")
                # Here you could integrate with Tesseract OCR if needed

            text += page_text

            # Log progress for large documents
            if page_num % 10 == 0 and page_num > 0:
                logger.info(f"Processed {page_num} pages of {len(doc)}")

        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return text


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Split text into smaller chunks with improved semantic boundary respect
    """
    try:
        # Try to use nltk for better sentence boundaries if available
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            has_nltk = True
        except ImportError:
            has_nltk = False

        if has_nltk:
            # Split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)  # Better paragraph splitting
            chunks = []
            current_chunk = ""

            # For article detection
            article_pattern = re.compile(r'Article\s+(\d+)', re.IGNORECASE)

            for para in paragraphs:
                # Check if this paragraph starts a new article
                article_match = article_pattern.match(para.strip())

                # If adding this paragraph exceeds chunk size or we hit a new article
                if (len(current_chunk) + len(para) > chunk_size and current_chunk) or article_match:
                    # If we already have content and hit a new chunk boundary
                    if current_chunk:
                        chunks.append(current_chunk)

                        # Keep overlap by including last sentences for context
                        sentences = nltk.sent_tokenize(current_chunk)
                        overlap_text = " ".join(sentences[-3:]) if len(sentences) > 3 else ""

                        # If we hit a new article, start fresh to maintain article boundaries
                        if article_match:
                            current_chunk = para + "\n\n"
                        else:
                            current_chunk = overlap_text + "\n\n"
                    else:
                        current_chunk = para + "\n\n"
                else:
                    current_chunk += para + "\n\n"

            # Add the last chunk if not empty
            if current_chunk.strip():
                chunks.append(current_chunk)

            logger.info(f"Created {len(chunks)} semantic chunks from text")
            return chunks

        else:
            # Fall back to improved basic method if nltk not available
            chunks = []
            start = 0

            # Simple paragraph and sentence detection
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size:
                    # Finalize current chunk
                    chunks.append(current_chunk)

                    # Simple overlap - take the last ~100 chars
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]

                current_chunk += para + "\n\n"

            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)

            logger.info(f"Created {len(chunks)} basic chunks from text")
            return chunks

    except Exception as e:
        logger.error(f"Error in chunking text: {e}")
        # Fall back to a very simple chunking method in case of error
        simple_chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            simple_chunks.append(text[i:i + chunk_size])
        return simple_chunks


def store_document_and_chunks(title, text, file_path=None, language='fr'):
    """
    Process document text, create chunks and store in the database
    """
    global document_vectors, chunk_ids, chunks_df

    try:
        # Create the document
        document = Document.objects.create(
            title=title,
            content=text,
            file=file_path,
            language=language
        )

        # Chunk the text with improved chunking
        chunks = chunk_text(text)
        logger.info(f"Storing {len(chunks)} chunks for document: {title}")

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

        # Rebuild the index after adding new documents
        setup_index()

        return document

    except Exception as e:
        logger.error(f"Error storing document and chunks: {e}")
        raise


def setup_index():
    """
    Load all document chunks from the database and create an index
    Tries to use sentence transformers if available, falls back to TF-IDF
    """
    global document_vectors, chunk_ids, chunks_df, use_sentence_transformers

    try:
        # Get all document chunks
        chunks = DocumentChunk.objects.all()

        # If no chunks exist, return None
        if not chunks.exists():
            logger.warning("No document chunks found in database")
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

        # Use sentence transformers if available
        if use_sentence_transformers:
            try:
                document_vectors = embedding_model.encode(chunk_texts)
                logger.info(f"Created embeddings for {len(chunk_texts)} chunks using SentenceTransformer")
            except Exception as e:
                logger.error(f"Error creating embeddings with SentenceTransformer: {e}")
                use_sentence_transformers = False

        # Fall back to TF-IDF if sentence transformers fails or isn't available
        if not use_sentence_transformers:
            document_vectors = vectorizer.fit_transform(chunk_texts)
            logger.info(f"Created TF-IDF vectors for {len(chunk_texts)} chunks")

    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        document_vectors = None
        chunk_ids = []
        chunks_df = None


def find_relevant_chunks(query, top_k=5):
    """
    Find the most relevant document chunks for a given query with hybrid search
    """
    global document_vectors, chunk_ids, chunks_df, use_sentence_transformers

    # If vectors not initialized, do it now
    if document_vectors is None or chunks_df is None:
        setup_index()

    # If still no vectors, return empty list
    if document_vectors is None or chunks_df is None:
        logger.warning("No document vectors available for search")
        return []

    try:
        # Preprocess the query to handle typos
        processed_query = preprocess_query(query)

        # Extract intent keywords for better matching
        intent_keywords = extract_intent_keywords(processed_query)
        enriched_query = processed_query

        # Add intent keywords to query if found
        if intent_keywords:
            enriched_query = processed_query + " " + " ".join(intent_keywords)
            logger.info(f"Enriched query with keywords: {intent_keywords}")

        # Calculate similarity based on the method being used
        if use_sentence_transformers:
            # Transform query to embedding vector
            query_vector = embedding_model.encode([enriched_query])[0]

            # Calculate cosine similarity
            if isinstance(document_vectors, np.ndarray):
                similarities = cosine_similarity([query_vector], document_vectors)[0]
            else:
                logger.error("Document vectors in unexpected format")
                return []
        else:
            # Transform query to TF-IDF vector
            query_vector = vectorizer.transform([enriched_query])

            # Calculate similarity
            similarities = cosine_similarity(query_vector, document_vectors).flatten()

        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Get the corresponding chunk IDs
        relevant_chunk_ids = [chunk_ids[i] for i in top_indices]

        # Get the chunks from the database
        relevant_chunks = DocumentChunk.objects.filter(chunk_id__in=relevant_chunk_ids)

        # Log search results with similarity scores
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
        for i, idx in enumerate(top_indices):
            logger.info(f"  Chunk {i + 1}: {chunk_ids[idx]} (score: {similarities[idx]:.4f})")

        # Now try to find articles if the query seems to be asking about specific articles
        if re.search(r'\barticle\s+\d+\b', processed_query, re.IGNORECASE):
            article_num_match = re.search(r'\barticle\s+(\d+)\b', processed_query, re.IGNORECASE)
            if article_num_match:
                article_num = article_num_match.group(1)
                logger.info(f"Detected specific article request: Article {article_num}")

                # Search for chunks containing this article
                article_chunks = DocumentChunk.objects.filter(
                    content__icontains=f"Article {article_num}"
                )

                # Add these to the results if not already included
                article_chunk_ids = set(article_chunks.values_list('chunk_id', flat=True))
                existing_chunk_ids = set(relevant_chunks.values_list('chunk_id', flat=True))
                missing_chunk_ids = article_chunk_ids - existing_chunk_ids

                if missing_chunk_ids:
                    additional_chunks = DocumentChunk.objects.filter(chunk_id__in=missing_chunk_ids)
                    logger.info(f"Adding {additional_chunks.count()} article-specific chunks")
                    relevant_chunks = list(relevant_chunks) + list(additional_chunks)

        return relevant_chunks

    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        return []


def process_policy_document(pdf_path):
    """
    Process the NGO policy document with improved logging and extraction
    """
    try:
        logger.info(f"Processing policy document: {pdf_path}")

        # Extract text with better PDF handling
        text = extract_text_from_pdf(pdf_path)

        if not text:
            logger.error("Failed to extract text from PDF")
            raise ValueError("No text extracted from PDF")

        # Try to extract a better title from the document content
        title = "Décret-loi n° 2011-88 du 24 septembre 2011"  # Default title

        # Look for a better title in the first few lines
        lines = text.split('\n')
        for i in range(min(10, len(lines))):
            line = lines[i].strip()
            if re.search(r'(Décret|Loi|Code|Règlement)', line, re.IGNORECASE) and len(line) > 10:
                title = line
                break

        # Store document and create chunks with improved chunking
        document = store_document_and_chunks(
            title=title,
            text=text,
            file_path=pdf_path,
            language='fr'
        )

        # Initialize the index with new content
        setup_index()

        logger.info(f"Successfully processed document with ID: {document.id}")
        return document

    except Exception as e:
        logger.error(f"Failed to process policy document: {e}")
        raise


def analyze_document_structure(doc_id):
    """
    Analyze the structure of a document to identify articles, sections, etc.
    """
    try:
        document = Document.objects.get(id=doc_id)
        text = document.content

        # Find articles
        articles = {}

        # Regular expression to find articles
        article_pattern = re.compile(r'Article\s+(\d+)[.\s:–-]+([^A-Z]+?)(?=Article\s+\d+|\Z)', re.DOTALL)

        # Extract all articles
        matches = article_pattern.finditer(text)
        for match in matches:
            article_num = match.group(1)
            article_text = match.group(2).strip()
            articles[article_num] = article_text

        logger.info(f"Found {len(articles)} articles in document {doc_id}")

        # Return the structure analysis
        return {
            'document_id': doc_id,
            'title': document.title,
            'article_count': len(articles),
            'articles': articles
        }

    except Document.DoesNotExist:
        logger.error(f"Document with ID {doc_id} not found")
        return None
    except Exception as e:
        logger.error(f"Error analyzing document structure: {e}")
        return None