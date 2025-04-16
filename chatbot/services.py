import os
import logging
from django.conf import settings
from .models import Conversation, Message, DocumentChunk
from .utils import find_relevant_chunks

logger = logging.getLogger(__name__)


class ChatbotService:
    """Service for handling chatbot interactions with local LLM"""

    def __init__(self):
        # Don't load the model immediately
        self.llm = None
        self.llm_available = False
        self.model_loaded = False

    def _load_model_if_needed(self):
        """Lazy-load the model only when needed"""
        if self.model_loaded:
            return self.llm_available

        try:
            from llama_cpp import Llama
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'mistral-7b-instruct-v0.1.Q4_K_M.gguf')

            if os.path.exists(model_path):
                # Use more memory-efficient settings
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=512,  # Reduce context window
                    n_threads=2,  # Reduce thread count
                    n_batch=8,  # Smaller batch size
                    n_gpu_layers=0,  # No GPU layers
                    offload_kqv=True,  # Offload key/query/value tensors
                    use_mlock=False  # Don't lock memory
                )
                self.llm_available = True
                logger.info(f"LLM loaded successfully with memory-optimized settings")
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
        chunks_text = "\n\n".join([chunk.content for chunk in relevant_chunks])

        return f"""Tu es un assistant spécialisé dans la législation tunisienne sur les associations, basé sur le Décret-loi n° 2011-88 du 24 septembre 2011.

Utilise uniquement les informations contenues dans les extraits suivants:

{chunks_text}

Si la réponse n'est pas dans les extraits, dis-le clairement. N'invente pas d'informations.
Sois précis, professionnel et concis dans tes réponses en français."""

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
        """Generate a response using the local LLM"""
        if not self._load_model_if_needed():
            return self._generate_fallback_response(query)

        system_prompt = self._create_system_prompt(relevant_chunks)

        # Build messages for chat completion
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for message in conversation_history:
                messages.append(message)

        # Add the current query
        messages.append({"role": "user", "content": query})

        try:
            # Generate the response using the local LLM
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )

            generated_text = response["choices"][0]["message"]["content"]
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response with local LLM: {e}")
            return "Je suis désolé, j'ai rencontré une erreur lors de la génération de la réponse. Veuillez réessayer."

    def _generate_fallback_response(self, query):
        """
        Generate a basic rule-based response when LLM is not available
        """
        # Simple pattern matching for common questions
        query_lower = query.lower()

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

        else:
            return ("Je ne peux pas répondre à cette question sans accès au modèle de langage. Veuillez consulter "
                    "directement le Décret-loi n° 2011-88 du 24 septembre 2011 portant organisation des associations.")

    def process_message(self, conversation, query):
        """
        Process a user message and generate a response
        """
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

        # Generate the response
        response_text = self._generate_response_with_local_llm(
            query, relevant_chunks, conversation_history
        )

        # Save the assistant's response
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=response_text
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
            "content": response_text,
            "relevant_documents": [
                {
                    "title": chunk.document.title,
                    "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk in relevant_chunks
            ]
        }