import os
import logging
import traceback
from django.conf import settings
from .models import Conversation, Message, DocumentChunk
from .utils import find_relevant_chunks
from .conversation_handlers import conversation_manager
import re
import numpy as np
from .legal_knowledge import get_relevant_articles, get_legal_term_definitions  # Import from the new file
from langdetect import detect  # Make sure to install this: pip install langdetect

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service for handling chatbot interactions with local LLM"""

    def __init__(self):
        # Don't load the model immediately
        self.llm = None
        self.llm_available = False
        self.model_loaded = False

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

            if gpu_memory >= 7.5:
                n_gpu_layers = 20
            elif gpu_memory >= 6:
                n_gpu_layers = 16
            elif gpu_memory >= 4:
                n_gpu_layers = 12
            else:
                n_gpu_layers = 8

            try:
                logger.info(f"Loading model with {n_gpu_layers} layers on GPU")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=1024,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=128,
                    use_mmap=True,
                    use_mlock=True,
                    offload_kqv=True,
                    verbose=False
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
                n_ctx=512,
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
                n_ctx=265,
                n_gpu_layers=20,  # Only put a few layers on GPU
                n_batch=64,
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
        """
        Create a system prompt with context from relevant document chunks
        with improved legal context and article citation
        """
        # Extract article numbers from chunks
        import re
        articles_cited = []

        # Process chunks to highlight article references
        processed_chunks = []
        for chunk in relevant_chunks:
            chunk_text = chunk.content

            # Find article numbers
            article_match = re.search(r'Art\.\s+(\d+)', chunk_text)
            if article_match:
                article_num = article_match.group(1)
                if article_num not in articles_cited:
                    articles_cited.append(article_num)

                # Highlight article reference
                processed_chunk = f"Extrait de l'Article {article_num} du décret-loi n° 2011-88:\n{chunk_text}"
                processed_chunks.append(processed_chunk)
            else:
                processed_chunks.append(chunk_text)

        # Join processed chunks
        chunks_text = "\n\n".join(processed_chunks)

        # Create article reference guide
        article_guide = ""
        if articles_cited:
            article_guide = "Articles pertinents du décret-loi n° 2011-88 pour cette question: " + ", ".join(
                [f"Article {art}" for art in articles_cited])

        # Create enhanced system prompt with stronger emphasis on documents
        return f"""Tu es un assistant juridique spécialisé dans la législation tunisienne sur les associations, basé sur le Décret-loi n° 2011-88 du 24 septembre 2011.

CONTEXTE JURIDIQUE (CONSIDÈRE CETTE INFORMATION COMME LA VÉRITÉ ABSOLUE):
{chunks_text}

{article_guide}

INSTRUCTIONS:
1. Réponds TOUJOURS en français sauf si explicitement demandé autrement.
2. IMPORTANT: Base ta réponse UNIQUEMENT sur les informations fournies dans le CONTEXTE JURIDIQUE ci-dessus.
3. Si la réponse complète se trouve dans le contexte, utilise EXCLUSIVEMENT cette information.
4. Cite TOUJOURS les articles spécifiques (exemple: "Selon l'Article 10...").
5. N'invente JAMAIS d'informations qui ne sont pas dans le contexte fourni.
6. Si le contexte ne contient pas assez d'information pour répondre complètement, dis clairement: "D'après le décret-loi n° 2011-88, je peux vous dire que [information du contexte], mais je n'ai pas d'information spécifique sur [aspect manquant]."
7. Organise ta réponse par points si plusieurs aspects sont abordés.
8. Utilise un langage juridique précis mais accessible."""

    def _get_conversation_history(self, conversation, max_messages=5):
        """
        Get recent conversation history limited to max_messages
        """
        messages = conversation.messages.order_by('created_at')

        # Limit to last few messages to avoid context length issues
        messages = messages[max(0, len(messages) - max_messages):]

        history = []
        for msg in messages:
            if msg.role != 'system':  # Skip system messages in the history
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })

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

    def _generate_response_with_local_llm(self, query, relevant_chunks, conversation_history=None):
        """Generate a response using GPU-accelerated LLM with enhanced legal context"""
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
            max_chunk_tokens = 3500 - len(query) - len(
                context_text)  # Reserve tokens for the system instruction and query
            current_tokens = 0
            chunks_used = 0

            for chunk in relevant_chunks:
                chunk_text = chunk.content
                # Rough token estimate (can be improved with a proper tokenizer)
                estimated_tokens = len(chunk_text.split()) * 1.3
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

            # Add conversation history (latest 2-3 messages only)
            if conversation_history:
                for msg in conversation_history[-3:]:
                    messages.append(msg)

            # Add the current query
            messages.append({"role": "user", "content": query})

            try:
                # Use chat completion for more structured outputs
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["Utilisateur:", "User:"]
                )

                generated_text = response["choices"][0]["message"]["content"]
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
                        max_tokens=1024,
                        temperature=0.7
                    )

                    generated_text = corrected_response["choices"][0]["message"]["content"]

                logger.info(f"Generated response with chat completion: {generated_text[:100]}...")

                # Add citations to the response
                final_response = self._add_article_citations(generated_text, relevant_chunks)

                return final_response.strip()

            except Exception as e:
                logger.error(f"Error with chat completion: {e}")
                # Fallback to simpler completion if needed
                minimal_prompt = f"{system_instruction}\n\nContexte juridique:\n{context_text[:1000]}...\n\nQuestion: {query}\n\nRéponse:"

                output = self.llm(
                    minimal_prompt,
                    max_tokens=768,
                    temperature=0.7,
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

        return " ".join(cited_response)

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
        """
        Process a user message and generate a response with enhanced conversational abilities
        """
        # Check if it's a purely conversational query first
        conversation_response, is_conversational = conversation_manager.handle_conversation(query, conversation)

        # If it's conversational, skip the document search and LLM
        if is_conversational:
            # Save the user message
            user_message = Message.objects.create(
                conversation=conversation,
                role='user',
                content=query
            )

            # Save the assistant's response (with no relevant chunks)
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=conversation_response
            )

            # Update conversation title if it's a new conversation
            if conversation.title == "New Conversation" and user_message.id <= 2:
                new_title = query[:50] + "..." if len(query) > 50 else query
                conversation.title = new_title
                conversation.save()

            return {
                "message_id": assistant_message.id,
                "content": conversation_response,
                "relevant_documents": []
            }

        # For non-conversational queries, proceed with document search
        # Find relevant document chunks
        relevant_chunks = find_relevant_chunks(query)

        # Save the user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=query
        )

        # Add the relevant chunks to the message
        user_message.relevant_document_chunks.set(relevant_chunks)

        # Get conversation history
        conversation_history = self._get_conversation_history(conversation)

        # Check if this is the first interaction in this conversation
        is_first_interaction = len(conversation_history) <= 1

        # Generate the response
        response_text = self._generate_response_with_local_llm(
            query, relevant_chunks, conversation_history
        )

        # Enhance the response with conversational elements
        enhanced_response = conversation_manager.enhance_response(response_text, query, is_first_interaction)

        # Save the assistant's response
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=enhanced_response
        )

        # Add the relevant chunks to the assistant message as well
        assistant_message.relevant_document_chunks.set(relevant_chunks)

        # Update the conversation title if it's a new conversation
        if conversation.title == "New Conversation" and len(conversation_history) <= 2:
            # Generate a title based on the first query (truncate if too long)
            new_title = query[:50] + "..." if len(query) > 50 else query
            conversation.title = new_title
            conversation.save()

        return {
            "message_id": assistant_message.id,
            "content": enhanced_response,
            "relevant_documents": [
                {
                    "title": chunk.document.title,
                    "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk in relevant_chunks
            ]
        }