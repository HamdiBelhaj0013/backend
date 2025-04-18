"""
Conversation handlers for more fluid chatbot interactions.
This module extends the chatbot's capabilities beyond document-based responses.
"""

import random
import re
from datetime import datetime


class ConversationManager:
    """Manages conversational aspects of the chatbot interactions"""

    def __init__(self):
        # Track conversation state
        self.greeting_given = False
        self.intro_given = False
        self.capabilities_explained = False

        # Define conversation patterns
        self.patterns = {
            'greeting': r'\b(hello|hi|hey|bonjour|salut|good morning|good afternoon|good evening)\b',
            'how_are_you': r'\b(how are you|comment ça va|how\'s it going|comment vas-tu)\b',
            'thank_you': r'\b(thank you|thanks|merci|thank|thx)\b',
            'goodbye': r'\b(goodbye|bye|au revoir|à plus tard|see you|à bientôt)\b',
            'help': r'\b(help|aide|what can you do|que peux-tu faire|capabilities|fonctionnalités)\b',
            'identity': r'\b(who are you|qui es-tu|what are you|your name|ton nom)\b',
            'capabilities': r'\b(what can you (do|tell me)|what do you know|que sais-tu|about what|à propos de quoi)\b',
            'source': r'\b(source|reference|référence|how do you know|comment sais-tu)\b',
        }

        # Response templates for various conversation types
        self.responses = {
            'greeting': [
                "Bonjour! Comment puis-je vous aider aujourd'hui?",
                "Salut! Je suis à votre disposition. Que puis-je faire pour vous?",
                "Bonjour! Je suis votre assistant spécialisé dans la législation tunisienne sur les associations. Comment puis-je vous aider?",
                "Hello! I can assist you in French regarding Tunisian association law. How may I help you?"
            ],
            'how_are_you': [
                "Je suis là pour vous aider concernant la législation tunisienne sur les associations. Comment puis-je vous être utile?",
                "Merci de demander. Je suis prêt à répondre à vos questions sur le décret-loi n° 2011-88 concernant les associations en Tunisie.",
                "Je fonctionne parfaitement! Comment puis-je vous assister aujourd'hui?"
            ],
            'thank_you': [
                "Je vous en prie! Y a-t-il autre chose que je puisse faire pour vous?",
                "C'est avec plaisir. N'hésitez pas si vous avez d'autres questions.",
                "De rien! Je suis ici pour vous aider avec la législation tunisienne sur les associations."
            ],
            'goodbye': [
                "Au revoir! N'hésitez pas à revenir si vous avez d'autres questions.",
                "À bientôt! Je serai là si vous avez besoin d'informations sur les associations en Tunisie.",
                "Au revoir et bonne journée!"
            ],
            'help': [
                "Je suis un assistant spécialisé dans la législation tunisienne sur les associations, basé sur le décret-loi n° 2011-88 du 24 septembre 2011. Je peux vous aider avec des questions sur la création d'associations, les statuts, le financement, la dissolution, et plus encore.",
                "Je peux répondre à vos questions concernant le décret-loi n° 2011-88 qui régit les associations en Tunisie. Par exemple, vous pouvez me demander comment créer une association, quelles sont les exigences pour les statuts, ou comment gérer les finances d'une association."
            ],
            'identity': [
                "Je suis un assistant virtuel spécialisé dans la législation tunisienne sur les associations, particulièrement le décret-loi n° 2011-88 du 24 septembre 2011.",
                "Je suis votre guide pour comprendre la législation tunisienne sur les associations. Je me base principalement sur le décret-loi n° 2011-88."
            ],
            'capabilities': [
                "Je peux vous renseigner sur de nombreux aspects du décret-loi n° 2011-88 concernant les associations en Tunisie, notamment:\n\n- La création d'associations\n- Les statuts et leur contenu requis\n- Le financement et les ressources\n- Les obligations légales\n- La dissolution d'associations\n\nN'hésitez pas à me poser des questions spécifiques sur ces sujets!",
                "Je suis spécialisé dans la législation tunisienne des associations. Vous pouvez me demander:\n\n- Comment créer une association\n- Quelles informations inclure dans les statuts\n- Comment gérer les ressources et financement\n- Les conditions d'adhésion et les membres\n- Le processus de dissolution"
            ],
            'source': [
                "Mes réponses sont basées sur le décret-loi n° 2011-88 du 24 septembre 2011 portant organisation des associations en Tunisie. C'est la référence légale principale qui régit les associations en Tunisie.",
                "Je m'appuie sur le décret-loi n° 2011-88 du 24 septembre 2011, qui est le texte de loi principal concernant les associations en Tunisie."
            ],
            'fallback': [
                "Je ne suis pas sûr de comprendre votre question. Je suis spécialisé dans la législation tunisienne sur les associations. Pourriez-vous reformuler ou me poser une question plus spécifique sur ce sujet?",
                "Pardonnez-moi, mais je ne peux pas répondre à cette question. Je suis conçu pour répondre aux questions concernant la législation tunisienne sur les associations, selon le décret-loi n° 2011-88.",
                "Je ne peux pas répondre avec certitude à cette question car elle semble en dehors de mon domaine d'expertise. Je me spécialise dans le décret-loi n° 2011-88 sur les associations en Tunisie."
            ]
        }

    def handle_conversation(self, query, conversation=None):
        """
        Handle conversational aspects of the user query

        Args:
            query: The user's question
            conversation: Optional conversation object for context

        Returns:
            response: A response if conversational, None if domain-specific
            is_conversational: Boolean indicating if this was a conversational exchange
        """
        query_lower = query.lower()

        # Check if this is the first interaction in the conversation
        is_new_conversation = conversation and len(conversation.messages.all()) <= 1

        # Check for greetings
        if re.search(self.patterns['greeting'], query_lower):
            return random.choice(self.responses['greeting']), True

        # Check for how are you
        if re.search(self.patterns['how_are_you'], query_lower):
            return random.choice(self.responses['how_are_you']), True

        # Check for thank you
        if re.search(self.patterns['thank_you'], query_lower):
            return random.choice(self.responses['thank_you']), True

        # Check for goodbye
        if re.search(self.patterns['goodbye'], query_lower):
            return random.choice(self.responses['goodbye']), True

        # Check for help request
        if re.search(self.patterns['help'], query_lower):
            return random.choice(self.responses['help']), True

        # Check for identity question
        if re.search(self.patterns['identity'], query_lower):
            return random.choice(self.responses['identity']), True

        # Check for capabilities question
        if re.search(self.patterns['capabilities'], query_lower):
            return random.choice(self.responses['capabilities']), True

        # Check for source question
        if re.search(self.patterns['source'], query_lower):
            return random.choice(self.responses['source']), True

        # If it's a new conversation and none of the patterns matched, we'll handle this
        # in the domain-specific logic
        return None, False

    def get_greeting(self):
        """Returns a greeting appropriate for the time of day"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            greeting = "Bonjour! Je suis votre assistant pour la législation tunisienne sur les associations."
        elif 12 <= hour < 18:
            greeting = "Bon après-midi! Je suis votre assistant pour la législation tunisienne sur les associations."
        else:
            greeting = "Bonsoir! Je suis votre assistant pour la législation tunisienne sur les associations."

        return greeting

    def enhance_response(self, response, query, is_first_interaction=False):
        """
        Enhance the domain-specific response with conversational elements

        Args:
            response: The domain-specific response to enhance
            query: The original query
            is_first_interaction: Whether this is the first message in conversation

        Returns:
            An enhanced response with conversational elements
        """
        enhanced_response = response

        # Add a greeting for the first interaction
        if is_first_interaction:
            greeting = self.get_greeting()
            enhanced_response = f"{greeting}\n\n{enhanced_response}"

        # Detect if the response doesn't contain answer
        if "je ne peux pas répondre" in response.lower() or "je n'ai pas d'information" in response.lower():
            enhanced_response += "\n\nPuis-je vous aider avec autre chose concernant les associations en Tunisie?"

        # For shorter responses, add a follow-up question prompt
        if len(response.split()) < 30:
            enhanced_response += "\n\nAvez-vous d'autres questions sur ce sujet?"

        return enhanced_response


# Instantiate the conversation manager globally to maintain state
conversation_manager = ConversationManager()