import numpy as np
import faiss
import os
import pickle
import logging
from sentence_transformers import SentenceTransformer
from django.conf import settings
from .models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunk_ids = []
        self.embedding_size = 768
        self.index_path = os.path.join(settings.MEDIA_ROOT, 'vector_index')
        os.makedirs(self.index_path, exist_ok=True)

    def _load_model(self):
        """Load the embedding model with caching"""
        if self.model is None:
            logger.info("Loading multilingual embedding model...")
            # Great for French legal text
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')
            logger.info("Embedding model loaded successfully")
        return self.model

    def build_index(self):
        """Build FAISS index from all document chunks"""
        logger.info("Building vector index from document chunks")
        self._load_model()

        # Get all chunks
        chunks = DocumentChunk.objects.all()
        if not chunks.exists():
            logger.warning("No document chunks found to index")
            return False

        # Extract text and IDs
        texts = [chunk.content for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Create embeddings in batches to avoid CUDA memory issues
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        # Create and configure FAISS index
        index = faiss.IndexFlatIP(self.embedding_size)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vectors to index
        index.add(embeddings)
        self.index = index

        # Save index and metadata
        self._save_index()
        logger.info(f"Vector index built successfully with {len(self.chunk_ids)} chunks")
        return True

    def _save_index(self):
        """Save index and metadata to disk"""
        if self.index is None:
            return False

        # Save FAISS index
        index_file = os.path.join(self.index_path, 'faiss_index.bin')
        faiss.write_index(self.index, index_file)

        # Save chunk IDs mapping
        mapping_file = os.path.join(self.index_path, 'chunk_mapping.pkl')
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.chunk_ids, f)

        logger.info(f"Vector index saved to {self.index_path}")
        return True

    def load_index(self):
        """Load index from disk if it exists"""
        index_file = os.path.join(self.index_path, 'faiss_index.bin')
        mapping_file = os.path.join(self.index_path, 'chunk_mapping.pkl')

        if not (os.path.exists(index_file) and os.path.exists(mapping_file)):
            logger.warning("Vector index files not found, need to build index first")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)

            # Load mapping
            with open(mapping_file, 'rb') as f:
                self.chunk_ids = pickle.load(f)

            logger.info(f"Vector index loaded with {len(self.chunk_ids)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            return False

    def search(self, query, top_k=5):
        """Search for relevant chunks by semantic similarity"""
        # Make sure model and index are loaded
        self._load_model()

        if self.index is None:
            if not self.load_index():
                if not self.build_index():
                    logger.error("Could not build or load vector index")
                    return []

        # Encode query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        # Search index
        scores, indices = self.index.search(query_vector, top_k)

        # Get results with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_ids) and idx >= 0:
                results.append({
                    'chunk_id': self.chunk_ids[idx],
                    'score': float(scores[0][i])
                })

        logger.info(f"Vector search returned {len(results)} results for query: {query[:30]}...")
        return results