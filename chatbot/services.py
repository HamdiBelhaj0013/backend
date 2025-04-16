import os
import requests
from django.conf import settings
from .models import Conversation, Message, DocumentChunk
from .utils import find_relevant_chunks


class ChatbotService:
    """Service for handling chatbot interactions"""

    def __init__(self):
        # Initialize Hugging Face client if API token is available
        self.hf_api_token = os.environ.get('HUGGINGFACE_API_TOKEN', '')
        self.llm_available = bool(self.hf_api_token)
        # Set the model to use - BLOOM is good for French
        self.hf_model = "bigscience/bloom"

    # [Keep the _create_system_prompt and _get_conversation_history methods as they are]

    def _generate_response_with_huggingface(self, query, relevant_chunks, conversation_history=None):
        """
        Generate a response using Hugging Face's Inference API
        """
        if not self.llm_available:
            return "Je ne peux pas générer de réponse pour le moment car le service de langage n'est pas configuré. Veuillez contacter l'administrateur."

        system_prompt = self._create_system_prompt(relevant_chunks)

        # Build the prompt for Hugging Face
        prompt = system_prompt + "\n\n"

        # Add conversation history if available
        if conversation_history:
            for message in conversation_history:
                if message["role"] == "user":
                    prompt += f"Question: {message['content']}\n"
                elif message["role"] == "assistant":
                    prompt += f"Réponse: {message['content']}\n"

        # Add the current query
        prompt += f"Question: {query}\nRéponse:"

        try:
            # Call the Hugging Face Inference API
            API_URL = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {"Authorization": f"Bearer {self.hf_api_token}"}

            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_length": 500, "return_full_text": False}}
            )

            if response.status_code == 200:
                # Extract and return the generated text
                generated_text = response.json()[0].get("generated_text", "")
                return generated_text.strip()
            else:
                print(f"Error from Hugging Face API: {response.text}")
                return "Je suis désolé, j'ai rencontré une erreur lors de la génération de la réponse. Veuillez réessayer."

        except Exception as e:
            print(f"Error generating response with Hugging Face: {e}")
            return "Je suis désolé, j'ai rencontré une erreur lors de la génération de la réponse. Veuillez réessayer."

    # [Keep the _generate_fallback_response method as it is]

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
        if self.llm_available:
            response_text = self._generate_response_with_huggingface(
                query, relevant_chunks, conversation_history
            )
        else:
            response_text = self._generate_fallback_response(query)

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