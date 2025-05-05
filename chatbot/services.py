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
        """Lazy-load the model only when needed with better error handling"""
        if self.model_loaded:
            return self.llm_available

        try:
            from llama_cpp import Llama
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'mistral-7b-instruct-v0.1.Q4_K_M.gguf')

            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path} with optimized settings")

                try:
                    # Use more memory-efficient settings
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=512,  # Reduce context window
                        n_threads=2,  # Reduce thread count
                        n_batch=8,  # Smaller batch size
                        use_mlock=False  # Don't lock memory
                    )
                    self.llm_available = True
                    logger.info(f"LLM loaded successfully with memory-optimized settings")
                except Exception as e:
                    logger.error(f"Error loading LLM with initial settings: {e}")
                    logger.info("Trying with minimal settings...")

                    # Try with absolute minimal settings
                    try:
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=256,
                            n_threads=1,
                            n_batch=1
                        )
                        self.llm_available = True
                        logger.info("LLM loaded with minimal settings")
                    except Exception as e:
                        logger.error(f"Error loading LLM with minimal settings: {e}")
                        self.llm_available = False
            else:
                logger.warning(f"LLM model file not found at {model_path}")
                self.llm_available = False

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm_available = False

        self.model_loaded = True
        return self.llm_available

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
        """Generate a response using the local LLM with better error handling"""
        if not self._load_model_if_needed():
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(query)

        try:
            # Create a simple prompt instead of chat format for testing
            chunks_text = "\n\n".join([chunk.content for chunk in relevant_chunks]) if relevant_chunks else ""

            simple_prompt = f"""Sur base des extraits du décret-loi n° 2011-88 du 24 septembre 2011 portant organisation des associations en Tunisie ci-dessous:

    {chunks_text}

    Question: {query}

    Réponse:"""

            logger.info(f"Generating response for query: {query}")
            logger.info(f"Using {len(relevant_chunks)} relevant chunks")

            # Try simple completion first
            try:
                logger.info("Attempting simple text completion")
                output = self.llm(
                    simple_prompt,
                    max_tokens=256,
                    temperature=0.7,
                    stop=["Question:", "\n\n"]
                )
                response_text = output['choices'][0]['text'].strip()
                logger.info(f"Generated response with simple completion: {response_text[:100]}...")
                return response_text
            except Exception as e:
                logger.error(f"Error with simple completion: {e}")

                # Fall back to original approach
                logger.info("Falling back to chat completion")
                system_prompt = self._create_system_prompt(relevant_chunks)

                # Build messages for chat completion
                messages = [{"role": "system", "content": system_prompt}]

                # Keep history minimal
                if conversation_history:
                    # Just add the last message if available
                    if len(conversation_history) > 0:
                        messages.append(conversation_history[-1])

                # Add the current query
                messages.append({"role": "user", "content": query})

                # Try chat completion
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=256,  # Reduced from 512
                    temperature=0.7
                )

                generated_text = response["choices"][0]["message"]["content"]
                logger.info(f"Generated response with chat completion: {generated_text[:100]}...")
                return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response with local LLM: {str(e)}")
            logger.error(traceback.format_exc())

            # Try with a hardcoded response based on the query
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