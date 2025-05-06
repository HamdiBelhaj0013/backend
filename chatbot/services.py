import os
import logging
import traceback
from django.conf import settings
from .models import Conversation, Message, DocumentChunk
from .utils import find_relevant_chunks
from .conversation_handlers import conversation_manager

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
        """Load model with optimized GPU acceleration"""
        try:
            import torch
            import os
            from llama_cpp import Llama

            # Force CUDA settings
            os.environ["LLAMA_CUBLAS"] = "1"
            os.environ["GGML_CUDA_FORCE_MMQ"] = "1"  # Force matrix multiplication on GPU

            # Model path
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
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB")

            # For RTX 4060 8GB, we can safely use more layers on GPU
            # Start with conservative settings and gradually increase if stable
            n_gpu_layers = 16  # Half the layers on GPU (total is 32)

            try:
                logger.info(f"Loading model with {n_gpu_layers} layers on GPU")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=512,  # Increase batch size for better throughput
                    use_mmap=False,  # Critical for GPU usage
                    use_mlock=True,
                    offload_kqv=True,  # Helps with memory efficiency
                    verbose=True
                )

                # Test if model loaded correctly
                test_result = self.llm("Test prompt.", max_tokens=5)
                logger.info(f"GPU model test successful: {test_result}")

                # Check VRAM usage after loading
                if hasattr(torch.cuda, 'memory_allocated'):
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(f"VRAM used after model load: {vram_used:.2f} GB")

                self.llm_available = True
                self.model_loaded = True
                return True

            except Exception as e:
                logger.error(f"Error with full GPU settings: {str(e)}")
                logger.error(traceback.format_exc())

                # Try with fewer GPU layers as fallback
                logger.info("Attempting with fewer GPU layers")
                try:
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_gpu_layers=8,  # Reduced GPU layers
                        n_batch=256,
                        use_mmap=False,
                        use_mlock=True,
                        verbose=True
                    )
                    test_result = self.llm("Test prompt.", max_tokens=5)
                    logger.info(f"Reduced GPU model test successful: {test_result}")
                    self.llm_available = True
                    self.model_loaded = True
                    return True
                except:
                    # As a last resort, try with minimal GPU usage
                    return self._try_minimal_gpu(model_path)

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
        """
        # Get content from chunks or use empty string if no chunks
        chunks_text = "\n\n".join([chunk.content for chunk in relevant_chunks]) if relevant_chunks else ""

        # Create a more comprehensive system prompt
        return f"""Tu es un assistant IA convivial et serviable spécialisé dans la législation tunisienne sur les associations, basé sur le Décret-loi n° 2011-88 du 24 septembre 2011.

CONTEXTE:
{chunks_text}

INSTRUCTIONS:
1. Réponds toujours en français sauf si explicitement demandé autrement.
2. Sois amical, patient et serviable dans tes réponses.
3. Si la réponse se trouve dans le contexte fourni, utilise cette information.
4. Si la question est hors sujet, explique poliment que tu es spécialisé dans la législation tunisienne sur les associations.
5. Si la réponse n'est pas dans les extraits ou si tu n'es pas sûr, dis-le clairement sans inventer d'informations.
6. Tu peux répondre à des questions générales et participer à des conversations normales en plus des questions spécifiques sur les associations.
7. Sois précis, professionnel et concis dans tes réponses.
8. Cite les articles pertinents du décret-loi quand c'est approprié."""

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

    def _generate_response_with_local_llm(self, query, relevant_chunks, conversation_history=None):
        """Generate a response using GPU-accelerated LLM"""
        if not self._load_model_if_needed():
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(query)

        try:
            # Sort chunks by relevance score
            # For this to work, modify find_relevant_chunks to return chunks with scores

            # Construct a prompt with adaptive context
            system_instruction = "Tu es un expert juridique spécialisé dans la législation tunisienne sur les associations."
            context_text = ""

            # Determine how many chunks we can use based on query length
            max_chunk_tokens = 3500 - len(query)  # Reserve ~500 tokens for the system instruction and query
            current_tokens = 0
            chunks_used = 0

            for chunk in relevant_chunks:
                chunk_text = chunk.content
                # Rough token estimate (can be improved with a proper tokenizer)
                estimated_tokens = len(chunk_text.split()) * 1.3
                if current_tokens + estimated_tokens > max_chunk_tokens:
                    break

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
                    max_tokens=1024,  # Generate longer responses
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,  # Reduce repetition
                    stop=["Utilisateur:", "User:"]  # Stop at appropriate boundaries
                )

                generated_text = response["choices"][0]["message"]["content"]
                logger.info(f"Generated response with chat completion: {generated_text[:100]}...")
                return generated_text.strip()

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