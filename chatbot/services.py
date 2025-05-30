import os
import logging
import traceback
from django.conf import settings
from .models import Conversation, Message, DocumentChunk
from .utils import find_relevant_chunks
from .conversation_handlers import conversation_manager
import re
import numpy as np
from functools import lru_cache
import time
import torch
from .legal_knowledge import get_relevant_articles, get_legal_term_definitions, LEGAL_TERMS
from langdetect import detect

logger = logging.getLogger(__name__)

# Create a singleton instance for the application
_GLOBAL_CHATBOT_SERVICE = None


def get_chatbot_service():
    """Get or create the global chatbot service instance"""
    global _GLOBAL_CHATBOT_SERVICE
    if _GLOBAL_CHATBOT_SERVICE is None:
        _GLOBAL_CHATBOT_SERVICE = ChatbotService()
        # Pre-load the model at startup
        _GLOBAL_CHATBOT_SERVICE._load_model_with_enhanced_gpu()
    return _GLOBAL_CHATBOT_SERVICE


class ChatbotService:
    """Service for handling chatbot interactions with local LLM"""

    def __init__(self):
        # Don't load the model immediately
        self.llm = None
        self.llm_available = False
        self.model_loaded = False
        self.recent_queries_cache = {}  # Cache for storing recent query results

        # Try to load sentence transformer for semantic similarity if available
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            self.semantic_model_available = True
            logger.info("Sentence transformer model loaded successfully for semantic matching")
        except:
            self.semantic_model_available = False
            logger.warning("Sentence transformer not available, falling back to keyword matching")

    def _load_model_if_needed(self):
        """Lazy-load the model with optimized GPU acceleration"""
        if self.model_loaded:
            return self.llm_available

        return self._load_model_with_enhanced_gpu()

    def _load_model_with_enhanced_gpu(self):
        try:
            import torch
            import os
            from llama_cpp import Llama

            # Force CUDA settings
            os.environ["LLAMA_CUBLAS"] = "1"
            os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force using the first GPU

            model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'mistral-7b-instruct-v0.1.Q4_K_M.gguf')
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return False

            # Check GPU
            gpu_available = torch.cuda.is_available()
            if not gpu_available:
                logger.warning("No CUDA-compatible GPU detected, using CPU")
                return self._fallback_to_cpu(model_path)

            # Get GPU info
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB")

            try:
                logger.info(f"Loading model with optimized settings on GPU")
                # Enhanced model loading with optimized settings
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,  # Double context window (was 2048)
                    n_gpu_layers=-1,  # Use all available layers for max quality
                    n_batch=512,  # Increased from 128 for better throughput
                    use_mmap=True,
                    use_mlock=True,
                    offload_kqv=True,
                    seed=42,  # Set consistent seed for more predictable outputs
                    verbose=False,
                    rope_freq_scale=0.5,  # Better handling of longer contexts
                    rope_scaling_type=1  # Linear scaling for improved long-text reasoning
                )

                # Test if model loaded correctly with a small prompt
                test_result = self.llm("Test prompt.", max_tokens=5)
                logger.info(f"GPU model test successful: {test_result}")

                # Monitor VRAM usage
                if hasattr(torch.cuda, 'memory_allocated'):
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(f"VRAM used after model load: {vram_used:.2f} GB")

                    # Release CUDA cache if needed
                    if vram_used > gpu_memory * 0.9:  # If using >90% of memory
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache to reduce memory usage")

                self.llm_available = True
                self.model_loaded = True
                return True

            except Exception as e:
                logger.error(f"Error with GPU settings: {str(e)}")
                logger.error(traceback.format_exc())
                return self._try_reduced_gpu_settings(model_path)

        except Exception as e:
            logger.error(f"Unexpected error in GPU initialization: {str(e)}")
            logger.error(traceback.format_exc())
            return self._fallback_to_cpu(model_path)

    def _try_minimal_gpu(self, model_path):
        """Try with absolute minimal GPU settings"""
        try:
            from llama_cpp import Llama
            logger.info("Attempting with minimal GPU settings")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=1024,
                n_gpu_layers=1,  # Absolute minimum
                n_batch=64,
                use_mmap=False,
                use_mlock=True,
                verbose=True
            )
            test_result = self.llm("Test.", max_tokens=5)
            logger.info(f"Minimal GPU model test successful: {test_result}")
            self.llm_available = True
            self.model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error with minimal GPU settings: {str(e)}")
            return self._fallback_to_cpu(model_path)

    def _try_reduced_gpu_settings(self, model_path):
        """Try loading with minimal GPU usage as a fallback"""
        try:
            from llama_cpp import Llama
            logger.info("Attempting to load with reduced GPU settings...")

            # Very conservative settings that should work on most GPUs
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=20,  # Only put a few layers on GPU
                n_batch=128,
                use_mlock=True,
                offload_kqv=True,
                seed=42,
                verbose=True  # Enable verbose to see more diagnostics
            )
            self.llm_available = True
            logger.info("LLM loaded with minimal GPU acceleration")
            return True
        except Exception as e:
            logger.error(f"Error loading LLM with reduced GPU settings: {str(e)}")
            return False

    def diagnose_gpu_issues(self):
        """Run diagnostics to identify GPU-related issues"""
        try:
            import torch
            import sys
            import subprocess

            results = {
                "pytorch_available": True,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": 0,
                "gpu_details": [],
                "cuda_version": None,
                "system_info": {}
            }

            # Check CUDA version
            if hasattr(torch.version, 'cuda'):
                results["cuda_version"] = torch.version.cuda

            # Get GPU details
            if results["cuda_available"]:
                results["gpu_count"] = torch.cuda.device_count()
                for i in range(results["gpu_count"]):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        results["gpu_details"].append({
                            "index": i,
                            "name": props.name,
                            "compute_capability": f"{props.major}.{props.minor}",
                            "memory_gb": props.total_memory / (1024 ** 3)
                        })
                    except Exception as e:
                        results["gpu_details"].append({
                            "index": i,
                            "error": str(e)
                        })

            # Check if nvidia-smi is available (more reliable GPU detection)
            try:
                nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
                results["nvidia_smi_output"] = nvidia_smi.stdout
            except:
                results["nvidia_smi_output"] = "nvidia-smi not available or failed"

            # Test CUDA with a simple operation
            if results["cuda_available"]:
                try:
                    x = torch.cuda.FloatTensor([1.0, 2.0])
                    y = x + x
                    results["cuda_test"] = "Success"
                    del x, y  # Clean up
                except Exception as e:
                    results["cuda_test"] = f"Failed: {str(e)}"

            logger.info(f"GPU Diagnostics: {results}")
            return results
        except Exception as e:
            logger.error(f"Error running GPU diagnostics: {str(e)}")
            return {"error": str(e)}

    def _fallback_to_cpu(self, model_path):
        """Fall back to CPU-only mode with optimized settings"""
        try:
            from llama_cpp import Llama
            import os

            # Get available CPU resources
            cpu_count = os.cpu_count() or 4

            logger.info(f"Falling back to CPU-only mode with {cpu_count} cores")

            # Optimize for CPU usage
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=32,
                n_threads=max(1, cpu_count - 1),  # Leave one core free for system
                use_mlock=True,
                seed=42,
                verbose=False
            )
            self.llm_available = True
            logger.info("LLM loaded successfully with CPU-only acceleration")
            return True
        except Exception as e:
            logger.error(f"Error loading LLM with CPU fallback: {str(e)}")
            self.llm_available = False
            return False

    def _create_system_prompt(self, relevant_chunks):
        """Create an enhanced system prompt with better legal context"""
        # Extract article numbers and structure information better
        articles_cited = []
        processed_chunks = []

        for chunk in relevant_chunks:
            chunk_text = chunk.content
            article_match = re.search(r'Art\.\s+(\d+)', chunk_text)
            if article_match:
                article_num = article_match.group(1)
                if article_num not in articles_cited:
                    articles_cited.append(article_num)

                # Highlight and structure the legal content
                processed_chunk = f"Extrait de l'Article {article_num}:\n{chunk_text}"
                processed_chunks.append(processed_chunk)
            else:
                processed_chunks.append(chunk_text)

        # Create a more structured prompt with clear sections
        chunks_text = "\n\n".join(processed_chunks)

        # Add article reference guide
        article_guide = ""
        if articles_cited:
            article_guide = "Articles pertinents du décret-loi n° 2011-88: " + ", ".join(
                [f"Article {art}" for art in articles_cited])

        return f"""Tu es un expert juridique spécialisé dans la législation tunisienne sur les associations, basé sur le Décret-loi n° 2011-88 du 24 septembre 2011.

CONTEXTE JURIDIQUE (VÉRITÉ ABSOLUE):
{chunks_text}

{article_guide}

INSTRUCTIONS:
1. Réponds TOUJOURS en français avec un style juridique clair et accessible.
2. Structure ta réponse: d'abord les principes généraux, puis les détails spécifiques.
3. Cite SYSTÉMATIQUEMENT les articles (ex: "Selon l'Article 10 du décret-loi n° 2011-88...").
4. Utilise des PUCES ou NUMÉROS pour les listes d'exigences.
5. Si l'information est incomplète, indique clairement ce que tu sais et ce qui manque.
6. RAPPELLE-TOI DES QUESTIONS PRÉCÉDENTES quand tu réponds aux nouvelles questions.
7. Si la réponse implique des articles spécifiques non mentionnés dans le contexte, indique: "D'autres articles du décret-loi peuvent contenir des informations supplémentaires."
"""

    def _extract_keywords(self, text):
        """Extract important keywords from text, focused on legal terminology"""
        # Convert to lowercase and tokenize
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Remove common French stopwords
        stopwords = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'en', 'du',
                     'de', 'à', 'au', 'aux', 'ce', 'ces', 'dans', 'par', 'pour',
                     'sur', 'que', 'qui', 'quoi', 'comment', 'est', 'sont', 'être']
        words = [word for word in words if word not in stopwords]

        # Check for legal terminology from our legal_knowledge module
        legal_keywords = []
        for word in words:
            if len(word) > 3:  # Only consider words with >3 characters
                for concept in LEGAL_TERMS:
                    if word in concept.lower():
                        legal_keywords.append(word)
                        break

        # Add any domain-specific keywords not covered by LEGAL_TERMS
        domain_keywords = ['association', 'décret', 'loi', 'statut', 'article', 'membre',
                           'adhésion', 'financement', 'dissolution', 'création', 'constituer']

        for word in words:
            if word in domain_keywords and word not in legal_keywords:
                legal_keywords.append(word)

        # If we couldn't find any relevant keywords, just return original words
        return legal_keywords if legal_keywords else words[:10]

    def _semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        # If sentence transformer model is available, use it
        if hasattr(self, 'semantic_model_available') and self.semantic_model_available:
            try:
                # Get embeddings
                embedding1 = self.sentence_model.encode(text1)
                embedding2 = self.sentence_model.encode(text2)

                # Normalize vectors
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)

                if norm1 > 0 and norm2 > 0:
                    # Calculate cosine similarity
                    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                    return float(similarity)
                return 0.0
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {str(e)}")
                return self._calculate_fallback_similarity(text1, text2)
        else:
            # Fall back to simpler method
            return self._calculate_fallback_similarity(text1, text2)

    def _calculate_fallback_similarity(self, text1, text2):
        """Calculate text similarity using simple token overlap when semantic model unavailable"""
        # Convert to lowercase and tokenize
        tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))

        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_similarity(self, query1, query2):
        """Calculate similarity between two queries for caching purposes"""
        # For cache comparisons, use a combination of exact match and semantic similarity
        if query1 == query2:
            return 1.0

        # First check keywords
        keywords1 = self._extract_keywords(query1)
        keywords2 = self._extract_keywords(query2)

        # Calculate keyword overlap (Jaccard similarity)
        keyword_set1 = set(keywords1)
        keyword_set2 = set(keywords2)
        keyword_intersection = keyword_set1.intersection(keyword_set2)
        keyword_union = keyword_set1.union(keyword_set2)

        if not keyword_union:
            return 0.0

        keyword_similarity = len(keyword_intersection) / len(keyword_union)

        # Add semantic similarity if available
        if hasattr(self, 'semantic_model_available') and self.semantic_model_available:
            semantic_sim = self._semantic_similarity(query1, query2)
            # Weight: 60% semantic, 40% keyword
            return 0.6 * semantic_sim + 0.4 * keyword_similarity
        else:
            return keyword_similarity

    def _summarize_messages(self, messages):
        """Summarize a sequence of conversation messages"""
        if not messages:
            return ""

        # Extract content from messages
        conversation_text = []
        for msg in messages:
            prefix = "User: " if msg.role == 'user' else "Assistant: "
            conversation_text.append(f"{prefix}{msg.content}")

        conversation_content = "\n".join(conversation_text)

        # If we have the LLM loaded, use it for summarization
        if self.llm_available:
            try:
                prompt = f"""Summarize this conversation history in 2-3 sentences, focusing on the main topics and questions:

{conversation_content}

Summary:"""

                result = self.llm(prompt, max_tokens=150, temperature=0.3, stop=["User:", "Assistant:"])
                summary = result['choices'][0]['text'].strip()

                if summary:
                    return summary
            except Exception as e:
                logger.error(f"Error summarizing messages with LLM: {str(e)}")

        # Fallback - extract main topics from user messages
        user_messages = [msg.content for msg in messages if msg.role == 'user']

        # Simple extractive summary if LLM summarization failed
        if len(user_messages) > 3:
            return "Topics discussed: " + ", ".join(
                [m[:50] + "..." if len(m) > 50 else m for m in user_messages[-3:]]
            )
        else:
            return "Previous questions: " + "; ".join(
                [m[:50] + "..." if len(m) > 50 else m for m in user_messages]
            )

    def _get_conversation_history(self, conversation, max_messages=8):
        """Get expanded conversation history with context summarization"""
        messages = conversation.messages.order_by('created_at')

        # If conversation is long, include a summary of earlier messages
        if len(messages) > max_messages:
            earlier_messages = list(messages[:len(messages) - max_messages])
            summary = self._summarize_messages(earlier_messages)

            # Include both summary and recent messages
            recent_messages = list(messages[len(messages) - max_messages:])

            history = [{"role": "system", "content": f"Earlier conversation summary: {summary}"}]
            for msg in recent_messages:
                if msg.role != 'system':
                    history.append({"role": msg.role, "content": msg.content})
        else:
            # For shorter conversations, include all messages
            history = []
            for msg in messages:
                if msg.role != 'system':
                    history.append({"role": msg.role, "content": msg.content})

        return history

    def _is_response_in_french(self, text):
        """Simple heuristic to check if text is primarily in French"""
        try:
            # Try to detect the language
            lang = detect(text)
            return lang == 'fr'
        except:
            # If detection fails, check for common French words
            french_markers = ['je', 'vous', 'est', 'sont', 'pour', 'dans', 'avec', 'sur', 'et', 'ou']
            words = text.lower().split()
            french_word_count = sum(1 for word in words if word in french_markers)
            return french_word_count >= min(3, len(words) // 5)  # At least 3 French words or 20% French

    def _hash_query(self, query):
        """Enhanced query normalization for effective caching"""
        # More aggressive normalization
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        normalized = re.sub(r'[,.?!;:«»\'\"\(\)]', '', normalized)

        # Remove common French stop words
        for word in ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'pour', 'dans', 'sur']:
            normalized = re.sub(r'\b' + word + r'\b', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Check for high similarity with existing cached queries
        for cached_query in self.recent_queries_cache:
            similarity = self._calculate_similarity(normalized, cached_query)
            if similarity > 0.85:  # High similarity threshold
                logger.info(f"Cache hit: Found similar query with similarity {similarity:.2f}")
                return cached_query

        return normalized

    def _prioritize_chunks(self, chunks, query):
        """Score chunks by relevance, article presence, and information density"""
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)

        scored_chunks = []
        for chunk in chunks:
            # Base score (assume it's stored in the chunk or use 1.0 as default)
            base_score = getattr(chunk, 'similarity_score', 1.0)

            # Check for article references (higher priority)
            article_match = re.search(r'Art\.\s+(\d+)', chunk.content)
            article_boost = 1.0
            if article_match:
                article_num = article_match.group(1)
                article_boost = 1.3  # Regular boost for containing an article

                # Extra boost if the article number is explicitly mentioned in the query
                if f"article {article_num}" in query.lower() or f"art. {article_num}" in query.lower():
                    article_boost = 1.5  # Higher boost for relevant article

            # Calculate keyword matching
            keyword_matches = sum(1 for keyword in query_keywords if keyword in chunk.content.lower())
            keyword_score = min(1.0, keyword_matches * 0.15)

            # Calculate information density
            legal_term_count = sum(1 for term in LEGAL_TERMS if term.lower() in chunk.content.lower())
            density_boost = 1.0 + (legal_term_count * 0.05)  # 5% boost per legal term

            # Calculate semantic similarity if model available
            semantic_score = 0.5  # Default medium score
            if hasattr(self, 'semantic_model_available') and self.semantic_model_available:
                semantic_score = self._semantic_similarity(query, chunk.content)

            # Calculate final score with weights
            final_score = (
                    base_score * 0.2 +  # Original ranking (20%)
                    article_boost * 0.3 +  # Article presence (30%)
                    keyword_score * 0.2 +  # Keyword matching (20%)
                    semantic_score * 0.2 +  # Semantic similarity (20%)
                    density_boost * 0.1  # Information density (10%)
            )

            scored_chunks.append((chunk, final_score))

        # Return chunks sorted by score (highest first)
        sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        logger.info(f"Prioritized {len(chunks)} chunks, top score: {sorted_chunks[0][1]:.2f}")

        return [chunk for chunk, _ in sorted_chunks]

    @lru_cache(maxsize=100)
    def _cached_find_relevant_chunks(self, query_hash, top_k=10):
        """Cached wrapper around find_relevant_chunks"""
        return find_relevant_chunks(query_hash, top_k)

    def _generate_response_with_local_llm(self, query, relevant_chunks, conversation_history=None):
        """Generate a response using optimized LLM with enhanced legal context"""
        if not self._load_model_if_needed():
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(query)

        try:
            # Get domain-specific context
            relevant_articles = get_relevant_articles(query)
            legal_terms = get_legal_term_definitions(query)

            # Enhance chunk retrieval with domain knowledge
            if relevant_articles and len(relevant_chunks) < 3:
                # Try to find chunks containing relevant articles
                for article in relevant_articles:
                    article_chunks = DocumentChunk.objects.filter(content__contains=f"Art. {article}")
                    if article_chunks.exists():
                        # Add these chunks if they're not already in relevant_chunks
                        for chunk in article_chunks:
                            if chunk not in relevant_chunks:
                                relevant_chunks = list(relevant_chunks) + [chunk]
                                break  # Just add one chunk per article to avoid too much context

            # Prioritize chunks by relevance
            relevant_chunks = self._prioritize_chunks(relevant_chunks, query)

            # Construct a prompt with adaptive context
            system_instruction = "Tu es un expert juridique spécialisé dans la législation tunisienne sur les associations. IMPORTANT: Réponds TOUJOURS en français."
            context_text = ""

            # Add legal term definitions if relevant
            if legal_terms:
                context_text += "Définitions de termes juridiques pertinents:\n"
                for term, definition in legal_terms.items():
                    context_text += f"- {term}: {definition}\n"
                context_text += "\n"

            # Add relevant articles from domain knowledge
            if relevant_articles:
                context_text += f"Articles particulièrement pertinents pour cette question: {', '.join(['Article ' + art for art in relevant_articles])}\n\n"

            # Determine how many chunks we can use based on query length
            # Increased token budget for more comprehensive context
            max_chunk_tokens = 6000 - len(query) - len(context_text)
            current_tokens = 0
            chunks_used = 0

            for chunk in relevant_chunks:
                chunk_text = chunk.content
                # Better token estimation using character count
                estimated_tokens = len(chunk_text) / 4  # Approx. 4 chars per token
                if current_tokens + estimated_tokens > max_chunk_tokens:
                    break

                # Highlight article references
                article_match = re.search(r'Art\.\s+(\d+)', chunk_text)
                if article_match:
                    article_num = article_match.group(1)
                    context_text += f"\n\nExtrait de l'Article {article_num} du décret-loi n° 2011-88:\n{chunk_text}"
                else:
                    context_text += f"\n\nExtrait du document '{chunk.document.title}':\n{chunk_text}"

                current_tokens += estimated_tokens
                chunks_used += 1

            logger.info(f"Using {chunks_used} chunks out of {len(relevant_chunks)} available")

            # Construct chat format for better control
            messages = [
                {"role": "system", "content": f"{system_instruction}\n\nContexte juridique:\n{context_text}"},
            ]

            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Use last 6 messages
                    messages.append(msg)

            # Add the current query
            messages.append({"role": "user", "content": query})

            try:
                # Use enhanced chat completion for more structured outputs
                # Enhanced LLM parameters
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=2048,  # Increased token limit
                    temperature=0.3,  # Reduced from 0.5 for more precise legal answers
                    top_p=0.85,  # More focused token sampling
                    top_k=40,  # Slightly reduced for more precise responses
                    repeat_penalty=1.2,  # Increased to reduce repetition
                    presence_penalty=0.1,  # Reduced for more factual responses
                    frequency_penalty=0.3,  # Increased to reduce word repetition
                    stop=["Utilisateur:", "User:", "Question:"],
                    mirostat_mode=2,  # Enable adaptive sampling
                    mirostat_tau=4.0,  # Adjusted for better coherence
                    mirostat_eta=0.1  # Learning rate for adaptive sampling
                )

                generated_text = response["choices"][0]["message"]["content"]

                # Ensure response is in French
                if not self._is_response_in_french(generated_text):
                    # Add an additional system message to force French
                    corrected_messages = messages + [
                        {"role": "assistant", "content": generated_text},
                        {"role": "system",
                         "content": "Ta réponse doit être en français. Traduis la réponse précédente en français."}
                    ]

                    # Generate a new response
                    corrected_response = self.llm.create_chat_completion(
                        messages=corrected_messages,
                        max_tokens=2048,
                        temperature=0.5
                    )

                    generated_text = corrected_response["choices"][0]["message"]["content"]

                logger.info(f"Generated response with chat completion: {generated_text[:100]}...")

                # Clear CUDA cache after generation to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Add citations to the response
                final_response = self._add_article_citations(generated_text, relevant_chunks)

                return final_response.strip()

            except Exception as e:
                logger.error(f"Error with chat completion: {e}")
                # Fallback to simpler completion if needed
                minimal_prompt = f"{system_instruction}\n\nContexte juridique:\n{context_text[:1000]}...\n\nQuestion: {query}\n\nRéponse:"

                output = self.llm(
                    minimal_prompt,
                    max_tokens=1536,  # Increased from 768
                    temperature=0.6,
                    top_p=0.9,
                    stop=["Question:"]
                )
                response_text = output['choices'][0]['text'].strip()
                return response_text

        except Exception as e:
            logger.error(f"Error generating response with local LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_fallback_response(query)

    def _add_article_citations(self, response, relevant_chunks):
        """Add proper article citations to response text"""
        import re

        # Extract article numbers from chunks
        article_references = {}
        for chunk in relevant_chunks:
            article_match = re.search(r'Art\.\s+(\d+)', chunk.content)
            if article_match:
                article_num = article_match.group(1)
                article_references[article_num] = chunk.content

        # If no article references found, return the original response
        if not article_references:
            return response

        # Look for mentions of articles without proper citation
        sentences = re.split(r'(?<=[.!?])\s+', response)
        cited_response = []

        for sentence in sentences:
            # Check if sentence already contains a proper citation
            if re.search(r'article\s+\d+\s+du décret-loi', sentence.lower()):
                cited_response.append(sentence)
                continue

            # Look for article mentions
            article_mention = re.search(r'article\s+(\d+)', sentence.lower())
            if article_mention:
                article_num = article_mention.group(1)
                if article_num in article_references:
                    # Add full citation if not already present
                    if "décret-loi" not in sentence.lower():
                        cited_sentence = sentence.replace(f"article {article_num}",
                                                          f"article {article_num} du décret-loi n° 2011-88")
                        cited_response.append(cited_sentence)
                    else:
                        cited_response.append(sentence)
                else:
                    cited_response.append(sentence)
            else:
                cited_response.append(sentence)

        # Post-process to ensure consistent citation format
        result = " ".join(cited_response)
        # Standardize citation format
        result = re.sub(r'article (\d+) du décret-loi', r'Article \1 du décret-loi', result, flags=re.IGNORECASE)

        return result

    def _generate_fallback_response(self, query):
        """
        Generate a more versatile rule-based response when LLM is not available
        """
        # Simple pattern matching for common questions
        query_lower = query.lower()

        # Check if it's a conversational query first
        conversation_response, is_conversational = conversation_manager.handle_conversation(query)
        if is_conversational:
            return conversation_response

        if "créer" in query_lower and "association" in query_lower:
            return ("Pour créer une association en Tunisie, vous devez suivre le régime de déclaration selon "
                    "l'article 10 du décret-loi n° 2011-88. Envoyez une lettre recommandée au secrétaire général "
                    "du gouvernement avec les documents requis: déclaration, copies des cartes d'identité des "
                    "fondateurs, et deux exemplaires des statuts signés.")

        elif "statuts" in query_lower:
            return ("Les statuts d'une association doivent contenir: la dénomination officielle en arabe et en langue "
                    "étrangère le cas échéant, l'adresse du siège, les objectifs, les conditions d'adhésion, "
                    "l'organigramme, les modalités de prise de décision et le montant de la cotisation s'il existe.")

        elif "financement" in query_lower or "ressources" in query_lower:
            return (
                "Selon l'article 34, les ressources d'une association peuvent comprendre: les cotisations des membres, "
                "les aides publiques, les dons et legs d'origine nationale ou étrangère, et les recettes résultant de "
                "ses biens, activités et projets.")

        elif "dissolution" in query_lower:
            return (
                "La dissolution d'une association peut être volontaire (par décision de ses membres conformément aux "
                "statuts) ou judiciaire (par jugement du tribunal). En cas de dissolution, un liquidateur doit être désigné.")

        elif "adhésion" in query_lower or "membre" in query_lower:
            return (
                "Selon l'article 17 du décret-loi n° 2011-88, un membre d'une association doit être de nationalité tunisienne "
                "ou résident en Tunisie, avoir au moins 13 ans, accepter par écrit les statuts de l'association, et payer "
                "la cotisation si requise. Les fondateurs et dirigeants ne peuvent pas occuper des responsabilités dans des "
                "partis politiques (article 9).")

        else:
            # Generic response
            return ("Je ne suis pas en mesure de répondre à cette question spécifique pour le moment. "
                    "Je peux vous renseigner sur la création d'associations, les statuts, le financement, "
                    "la dissolution ou les conditions d'adhésion selon le décret-loi n° 2011-88. "
                    "Comment puis-je vous aider sur ces sujets?")

    def process_message(self, conversation, query):
        """Enhanced query processing with contextual awareness"""
        start_time = time.time()
        logger.info(f"Processing message: {query[:100]}...")

        # 1. Check if it's a conversational query first
        conversation_response, is_conversational = conversation_manager.handle_conversation(query, conversation)
        if is_conversational:
            return self._handle_conversational_response(conversation, query, conversation_response)

        # 2. Query preprocessing and normalization
        normalized_query = re.sub(r'\s+', ' ', query.strip())

        # 3. Enhanced caching with semantic similarity
        query_hash = self._hash_query(normalized_query)
        if query_hash in self.recent_queries_cache:
            cached_result = self.recent_queries_cache[query_hash]
            logger.info(f"Using cached result for similar query (saved {time.time() - start_time:.2f}s)")
            return self._handle_cached_response(conversation, query, cached_result)

        # 4. Advanced retrieval pipeline
        # 4.1 Get initial candidate chunks
        relevant_chunks = find_relevant_chunks(query, top_k=12)  # Get more candidates

        # 4.2 Extract legal concepts
        legal_concepts = get_relevant_articles(query)

        # 4.3 Retrieve additional context-specific chunks if needed
        if legal_concepts and len(relevant_chunks) < 3:
            context_chunks = []
            for article in legal_concepts:
                article_chunks = DocumentChunk.objects.filter(content__contains=f"Art. {article}")
                if article_chunks.exists():
                    context_chunks.extend(list(article_chunks[:2]))  # Add up to 2 chunks per article

            # Add these chunks if they're not already in relevant_chunks
            for chunk in context_chunks:
                if chunk not in relevant_chunks:
                    relevant_chunks = list(relevant_chunks) + [chunk]

        # 4.4 Re-rank all chunks with improved scoring
        relevant_chunks = self._prioritize_chunks(relevant_chunks, query)

        # 5. Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=query
        )
        user_message.relevant_document_chunks.set(relevant_chunks)

        # 6. Process conversation history with improved context
        conversation_history = self._get_conversation_history(conversation)
        is_first_interaction = len(conversation_history) <= 1

        # 7. Generate response with optimized LLM
        response_text = self._generate_response_with_local_llm(
            query, relevant_chunks, conversation_history
        )

        # 8. Enhance response with conversational elements
        enhanced_response = conversation_manager.enhance_response(response_text, query, is_first_interaction)

        # 9. Save assistant message
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=enhanced_response
        )
        assistant_message.relevant_document_chunks.set(relevant_chunks)

        # 10. Update conversation title if new
        if conversation.title == "New Conversation" and len(conversation_history) <= 2:
            new_title = query[:50] + "..." if len(query) > 50 else query
            conversation.title = new_title
            conversation.save()

        # 11. Update cache
        self.recent_queries_cache[query_hash] = {
            'response': enhanced_response,
            'chunks': relevant_chunks,
            'timestamp': time.time()
        }

        # 12. Prune cache if needed
        if len(self.recent_queries_cache) > 100:
            # Remove oldest entries
            cache_items = list(self.recent_queries_cache.items())
            sorted_by_time = sorted(cache_items, key=lambda x: x[1]['timestamp'])
            for i in range(10):  # Remove oldest 10 entries
                if i < len(sorted_by_time):
                    del self.recent_queries_cache[sorted_by_time[i][0]]

        logger.info(f"Processed message in {time.time() - start_time:.2f}s")

        # 13. Return formatted response
        return {
            "message_id": assistant_message.id,
            "content": enhanced_response,
            "relevant_documents": [
                {
                    "title": chunk.document.title,
                    "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk in relevant_chunks[:5]  # Limit to top 5 for response
            ]
        }

    def _handle_conversational_response(self, conversation, query, conversation_response):
        """Handle purely conversational messages"""
        # Save the user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=query
        )

        # Save the assistant's response
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=conversation_response
        )

        # Update conversation title if needed
        if conversation.title == "New Conversation" and user_message.id <= 2:
            new_title = query[:50] + "..." if len(query) > 50 else query
            conversation.title = new_title
            conversation.save()

        logger.info(f"Processed conversational message")
        return {
            "message_id": assistant_message.id,
            "content": conversation_response,
            "relevant_documents": []
        }

    def _handle_cached_response(self, conversation, query, cached_result):
        """Handle responses from cache"""
        # Save the user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=query
        )

        # Add relevant chunks from cache
        user_message.relevant_document_chunks.set(cached_result['chunks'])

        # Create assistant message with cached response
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=cached_result['response']
        )

        # Add the relevant chunks to the assistant message as well
        assistant_message.relevant_document_chunks.set(cached_result['chunks'])

        # Update conversation title if needed
        if conversation.title == "New Conversation" and len(conversation.messages.all()) <= 2:
            new_title = query[:50] + "..." if len(query) > 50 else query
            conversation.title = new_title
            conversation.save()

        return {
            "message_id": assistant_message.id,
            "content": cached_result['response'],
            "relevant_documents": [
                {
                    "title": chunk.document.title,
                    "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk in cached_result['chunks'][:5]  # Limit to top 5 for response
            ],
            "cached": True
        }

    def cleanup_resources(self):
        """Clean up resources when shutting down"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Release the model
        if self.llm:
            del self.llm
            self.llm = None
            self.model_loaded = False
            self.llm_available = False

        # Clear cache
        self.recent_queries_cache.clear()