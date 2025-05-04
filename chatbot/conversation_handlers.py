"""
Advanced conversation handlers for more fluid, context-aware chatbot interactions.
This module implements sophisticated conversation state tracking, sentiment analysis,
and dynamic response generation for a more human-like experience.
"""

import random
import re
import logging
import json
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List, Any, Set
import string
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class MemoryItem:
    """Represents a piece of information to remember about the conversation"""
    def __init__(self, key, value, importance=1, expiry=None):
        self.key = key
        self.value = value
        self.importance = importance  # 1-10 scale
        self.created_at = datetime.now()
        self.expiry = expiry  # When this memory should expire
        self.access_count = 0  # How often this memory has been accessed
        self.last_accessed = None

    def access(self):
        """Mark this memory as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def is_expired(self):
        """Check if this memory has expired"""
        if not self.expiry:
            return False
        return datetime.now() > self.expiry

    def relevance_score(self, current_time=None):
        """Calculate how relevant this memory is right now"""
        if not current_time:
            current_time = datetime.now()

        # Time decay - memories become less relevant as they age
        age_seconds = (current_time - self.created_at).total_seconds()
        time_factor = max(0, 1 - (age_seconds / (3600 * 24)))  # Decay over 24 hours

        # Access frequency increases relevance
        access_factor = min(1, self.access_count / 5)  # Max out at 5 accesses

        # Importance is inherent to the memory
        importance_factor = self.importance / 10

        # Calculate final score
        return (time_factor * 0.4) + (access_factor * 0.3) + (importance_factor * 0.3)

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'importance': self.importance,
            'created_at': self.created_at.isoformat(),
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data):
        """Create a MemoryItem from a dictionary"""
        item = cls(data['key'], data['value'], data['importance'])
        item.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('expiry'):
            item.expiry = datetime.fromisoformat(data['expiry'])
        item.access_count = data['access_count']
        if data.get('last_accessed'):
            item.last_accessed = datetime.fromisoformat(data['last_accessed'])
        return item

class ConversationMemory:
    """Manages the chatbot's memory of past interactions"""
    def __init__(self, max_items=100):
        self.memories = {}  # key -> MemoryItem
        self.max_items = max_items

    def remember(self, key, value, importance=1, expiry=None):
        """Store a new piece of information"""
        self.memories[key] = MemoryItem(key, value, importance, expiry)

        # If we're over capacity, forget the least important memories
        if len(self.memories) > self.max_items:
            self._forget_least_relevant()

    def recall(self, key):
        """Retrieve a memory by its key"""
        if key in self.memories:
            memory = self.memories[key]
            if memory.is_expired():
                del self.memories[key]
                return None

            memory.access()
            return memory.value
        return None

    def recall_relevant(self, keywords, max_items=5):
        """Retrieve the most relevant memories based on keywords"""
        relevant_memories = []

        # Convert keywords to a set for faster lookup
        if isinstance(keywords, str):
            keyword_set = set(keywords.lower().split())
        else:
            keyword_set = set(k.lower() for k in keywords)

        for memory in self.memories.values():
            # Skip expired memories
            if memory.is_expired():
                continue

            # Calculate keyword relevance
            key_str = memory.key.lower()
            val_str = str(memory.value).lower()

            # Check if any keywords are in the memory
            keyword_matches = sum(1 for k in keyword_set if k in key_str or k in val_str)
            keyword_score = keyword_matches / max(1, len(keyword_set))

            # Combine with general relevance
            total_score = (memory.relevance_score() * 0.7) + (keyword_score * 0.3)

            relevant_memories.append((memory, total_score))

        # Sort by relevance score and return the top N
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem.value for mem, score in relevant_memories[:max_items]]

    def _forget_least_relevant(self):
        """Remove the least relevant memories to make space"""
        now = datetime.now()

        # Calculate relevance for all memories
        memory_scores = [(memory, memory.relevance_score(now))
                         for memory in self.memories.values()]

        # Sort by relevance (lowest first)
        memory_scores.sort(key=lambda x: x[1])

        # Remove the 10% least relevant or at least one
        to_remove = max(1, len(memory_scores) // 10)
        for i in range(to_remove):
            if i < len(memory_scores):
                memory_to_remove = memory_scores[i][0]
                del self.memories[memory_to_remove.key]

    def to_dict(self):
        """Convert memory to dictionary for serialization"""
        return {
            'memories': {k: mem.to_dict() for k, mem in self.memories.items()},
            'max_items': self.max_items
        }

    @classmethod
    def from_dict(cls, data):
        """Create a ConversationMemory from a dictionary"""
        memory = cls(data.get('max_items', 100))
        for k, mem_data in data.get('memories', {}).items():
            memory.memories[k] = MemoryItem.from_dict(mem_data)
        return memory

class ConversationManager:
    """Advanced manager for conversational aspects of the chatbot interactions"""

    def __init__(self):
        # Enhanced conversation state tracking
        self.conversation_states = {}  # Stores state per conversation
        self.conversation_memories = {}  # Long-term memory per conversation

        # Entity recognition patterns
        self.entity_patterns = {
            'person': r'([A-Z][a-z]+ [A-Z][a-z]+|M\.\s[A-Z][a-z]+|Mme\.\s[A-Z][a-z]+)',
            'organization': r'([A-Z][a-z]* Association|Ministère de[s]? [A-Za-zéèêëàâäôöûüùïîç]+|Fondation [A-Za-z]+)',
            'location': r'(Tunis(?:ie)?|Sfax|Sousse|Kairouan|Bizerte|Gabès|Ariana|Gafsa)',
            'date': r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2} [a-zéû]+ \d{4})',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+216[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{3}|\d{2}[-\s]?\d{3}[-\s]?\d{3})',
            'amount': r'(\d+(?:[,.]\d+)?\s*(?:dinars?|TND|DT|euros?|EUR|€|dollars?|USD|\$))',
            'percentage': r'(\d+(?:[,.]\d+)?\s*%)',
            'association_type': r'\b(association|ONG|organisation|fondation)\b',
        }

        # Expanded conversation patterns with better typo handling
        self.patterns = {
            'greeting': r'\b(hello|hi|hey|bonjour|salut|bnjr|slt|bjr|good morning|good afternoon|good evening|coucou|hola|salam)\b',
            'how_are_you': r'\b(how are you|comment (ça|ca) va|how\'s it going|comment vas[\s-]tu|ca va|ça va|comment allez[\s-]vous)\b',
            'thank_you': r'\b(thank you|thanks|merci|thank|thx|thnks|mrci|mrc|thank u)\b',
            'goodbye': r'\b(goodbye|bye|au revoir|à plus tard|see you|à bientôt|salut|ciao|a\+|à \+|adieu)\b',
            'help': r'\b(help|aide|what can you do|que peux-tu faire|capabilities|fonctionnalités|fonctionalites|comment m\'aider)\b',
            'identity': r'\b(who are you|qui es-tu|what are you|your name|ton nom|t\'es qui|tu es qui|c\'est quoi ton nom|qui êtes[\s-]vous)\b',
            'capabilities': r'\b(what can you (do|tell me)|what do you know|que sais-tu|about what|à propos de quoi|sur quoi|tu connais quoi)\b',
            'source': r'\b(source|reference|référence|how do you know|comment sais-tu|d\'où vient|d\'ou vient|comment tu sais)\b',
            'clarification': r'\b(que veux[\s-]tu dire|what do you mean|explain|explique|peux[\s-]tu clarifier|clarify|elaborate|précise)\b',
            'opinion': r'\b(what do you think|que penses[\s-]tu|your opinion|ton avis|selon toi|d\'après toi)\b',
            'small_talk': r'\b(weather|temps|météo|favorite|préféré|hobby|loisir|family|famille|weekend|week[\s-]end|holiday|vacances)\b',
            'frustration': r'\b(not helpful|pas utile|c\'est faux|wrong|incorrect|useless|inutile|tu comprends pas|you don\'t understand)\b',
            'satisfaction': r'\b(perfect|parfait|great|super|génial|excellent|awesome|good|bien|très bien)\b',
            'repeat': r'\b(repeat|répéte|redis|say again|dis-moi encore|once more|rephrases)\b',
            'continue': r'\b(continue|go on|poursuis|et ensuite|et après|and then|next)\b',
            'previous': r'\b(previous|avant|go back|retourner|précédent)\b',
            'correction': r'\b(non|no|incorrect|ce n\'est pas|je voulais dire|i meant|i meant to say|je voulais|je ne|ce n\'est pas ça)\b'
        }

        # More sophisticated response templates with variability
        self.responses = self._initialize_response_templates()

        # Context-aware follow-up questions
        self.follow_ups = self._initialize_follow_ups()

        # Sentiment analysis words in French
        self.sentiment_words = {
            'positive': ['bon', 'bien', 'super', 'génial', 'excellent', 'parfait', 'merci', 'content', 'heureux', 'satisfait',
                        'facile', 'clair', 'utile', 'helpful', 'agréable', 'fantastique', 'formidable',
                        'impressionnant', 'incroyable', 'précis', 'professionnel', 'rapide', 'efficace'],
            'negative': ['mauvais', 'pas bien', 'pas bon', 'terrible', 'horrible', 'difficile', 'confus', 'faux', 'incorrect',
                        'inutile', 'lent', 'compliqué', 'frustrant', 'énervant', 'agaçant', 'déçu', 'décevant',
                        'insatisfait', 'non', 'problème', 'erreur', 'bug', 'plantage']
        }

        # Conversation topic transitions for more natural flow
        self.topic_transitions = {
            'creation': ['funding', 'legal', 'membership'],
            'funding': ['reporting', 'legal', 'administration'],
            'legal': ['creation', 'dissolution', 'obligations'],
            'dissolution': ['legal', 'reporting', 'obligations'],
            'membership': ['administration', 'rights', 'obligations'],
            'administration': ['membership', 'meetings', 'reporting'],
            'reporting': ['funding', 'legal', 'administration'],
            'obligations': ['legal', 'reporting', 'rights'],
            'rights': ['obligations', 'membership', 'legal'],
            'meetings': ['administration', 'membership', 'voting']
        }

    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize expanded response templates with more natural variations"""
        return {
            'greeting': [
                "Bonjour! Comment puis-je vous aider aujourd'hui concernant les associations en Tunisie?",
                "Salut! Je suis à votre disposition. Que puis-je faire pour vous?",
                "Bonjour! Comment puis-je vous assister avec vos questions sur la législation des associations?",
                "Bonjour et bienvenue! En quoi puis-je vous être utile aujourd'hui?",
                "Salut! Heureux de vous rencontrer. Avez-vous des questions sur la législation des associations?",
                "Bonjour! Je suis prêt à répondre à vos questions sur le décret-loi n° 2011-88. Comment puis-je vous aider?",
                "Bonjour! Que puis-je faire pour vous concernant la législation tunisienne des associations?",
                "Salam! Je suis votre assistant spécialisé dans le droit des associations en Tunisie. Comment puis-je vous aider?",
                "Bonjour et bienvenue! Je suis là pour répondre à toutes vos questions concernant la législation des associations. Que souhaitez-vous savoir?",
                "Bonjour! C'est un plaisir de vous aider avec vos questions concernant les associations en Tunisie. Par où voulez-vous commencer?"
            ],
            'how_are_you': [
                "Je suis là pour vous aider concernant la législation tunisienne sur les associations. Comment puis-je vous être utile aujourd'hui?",
                "Merci de demander. Je suis toujours prêt à répondre à vos questions! En quoi puis-je vous assister?",
                "Je fonctionne parfaitement! Avez-vous des questions sur les associations en Tunisie?",
                "Très bien, merci! J'espère pouvoir vous aider avec vos questions juridiques aujourd'hui.",
                "Je suis à votre service et prêt à vous aider avec toutes vos questions sur les associations.",
                "Je suis opérationnel et à votre service! Comment puis-je vous aider concernant les associations?",
                "Très bien, merci de vous en soucier! Avez-vous des questions sur la législation des associations?",
                "Je vais bien, merci! Je suis là pour vous aider à naviguer dans la législation tunisienne des associations. Que voulez-vous savoir?",
                "Parfaitement bien! Toujours prêt à partager mes connaissances sur le décret-loi n° 2011-88. Quelle est votre question?",
                "En pleine forme et à votre disposition! Je suis spécialisé dans la législation des associations en Tunisie. Comment puis-je vous être utile?"
            ],
            'thank_you': [
                "Je vous en prie! Y a-t-il autre chose que je puisse faire pour vous?",
                "C'est avec plaisir. N'hésitez pas si vous avez d'autres questions.",
                "De rien! Je suis ici pour vous aider avec la législation tunisienne sur les associations.",
                "Tout le plaisir est pour moi! Avez-vous d'autres questions sur la législation?",
                "Je vous en prie. Je reste à votre disposition pour toute autre question.",
                "Avec plaisir! N'hésitez pas à me solliciter si vous avez besoin d'autres informations.",
                "C'est normal! Avez-vous d'autres questions concernant les associations?",
                "Je suis content d'avoir pu vous aider! N'hésitez pas si vous avez d'autres questions.",
                "Le plaisir est partagé! Je reste disponible si vous avez d'autres interrogations sur les associations tunisiennes.",
                "Je suis là pour ça! Si d'autres aspects de la législation vous intéressent, n'hésitez pas à me demander.",
                "C'était un plaisir de vous aider! Avez-vous d'autres questions sur le décret-loi n° 2011-88?",
                "À votre service! Si vous avez besoin d'éclaircissements supplémentaires, je suis toujours là."
            ],
            'help': [
                "Je suis un assistant spécialisé dans la législation tunisienne sur les associations, basé sur le décret-loi n° 2011-88 du 24 septembre 2011. Je peux vous aider avec des questions sur:\n\n1. La création d'associations\n2. Les statuts et la structure\n3. Le financement et la gestion\n4. Les obligations légales\n5. La dissolution\n\nQue souhaitez-vous explorer?",

                "Je peux répondre à vos questions sur le décret-loi n° 2011-88 qui régit les associations en Tunisie. Par exemple:\n\n- Comment créer une association?\n- Quelles sont les exigences pour les statuts?\n- Comment gérer les finances?\n- Quelles sont vos obligations légales?\n- Comment procéder à une dissolution?\n\nQu'aimeriez-vous savoir?",

                "En tant qu'assistant juridique spécialisé, je peux vous aider à comprendre:\n\n• Le processus de création d'une association\n• Les droits et obligations des membres\n• Les structures de gouvernance\n• Les sources de financement autorisées\n• Les procédures administratives obligatoires\n• Les conditions de dissolution\n\nAvez-vous une question spécifique?",

                "Je vous accompagne dans tous les aspects juridiques des associations en Tunisie:\n\n- Formalités de création et d'enregistrement\n- Rédaction des statuts conformes\n- Gestion financière et comptabilité\n- Obligations de transparence et reporting\n- Relations avec l'administration\n- Procédures de modification et dissolution\n\nComment puis-je vous assister aujourd'hui?"
            ],
            'frustration': [
                "Je comprends votre frustration. Essayons d'aborder votre question différemment. Pourriez-vous me préciser ce que vous recherchez exactement?",

                "Je suis désolé si ma réponse n'était pas celle que vous attendiez. Reformulons ensemble votre question pour que je puisse mieux vous aider.",

                "Pardonnez-moi si je n'ai pas bien saisi votre demande. Pourriez-vous l'exprimer autrement, peut-être avec plus de détails?",

                "Je m'excuse si ma réponse n'était pas satisfaisante. Dites-moi plus précisément ce que vous souhaitez savoir sur les associations, et je ferai de mon mieux pour vous répondre clairement.",

                "Je regrette que ma réponse ne soit pas à la hauteur de vos attentes. Pouvez-vous me guider en me précisant l'aspect spécifique qui vous intéresse dans la législation des associations?",

                "Merci pour votre retour. Pour mieux vous aider, pourriez-vous me dire quelle partie de ma réponse n'était pas claire ou complète?"
            ],
            'identity': [
                "Je suis un assistant virtuel spécialisé dans la législation tunisienne sur les associations, particulièrement le décret-loi n° 2011-88 du 24 septembre 2011. Je suis là pour répondre à vos questions sur la création, la gestion et la dissolution des associations en Tunisie.",

                "Je m'appelle Assistant des Associations, un conseiller virtuel conçu pour vous aider à naviguer dans la législation tunisienne des associations. Ma mission est de rendre cette législation plus accessible et compréhensible pour tous.",

                "Je suis votre guide juridique spécialisé dans le droit des associations en Tunisie. Je peux vous orienter à travers les différentes dispositions du décret-loi n° 2011-88 et vous aider à comprendre vos droits et obligations.",

                "Je suis un assistant numérique dédié à la législation des associations en Tunisie. Je possède une connaissance approfondie du décret-loi n° 2011-88 et je suis là pour répondre à toutes vos questions juridiques dans ce domaine."
            ],
            'clarification': [
                "Pour clarifier ce point important: ",
                "Permettez-moi d'expliquer cela plus clairement: ",
                "Pour être plus précis sur ce sujet: ",
                "Laissez-moi reformuler pour plus de clarté: ",
                "En d'autres termes, plus simplement: ",
                "Pour éviter toute confusion, précisons que: ",
                "Pour mieux comprendre ce concept: ",
                "Si je peux exprimer cela différemment: "
            ],
            'source': [
                "Mes informations proviennent principalement du décret-loi n° 2011-88 du 24 septembre 2011, qui régit les associations en Tunisie. Ce texte législatif constitue le cadre juridique de référence pour les associations tunisiennes.",

                "Je me base sur le décret-loi n° 2011-88 du 24 septembre 2011, relatif à l'organisation des associations en Tunisie. Ce texte a remplacé la loi de 1959 et a considérablement libéralisé le secteur associatif tunisien.",

                "Ma source principale est le décret-loi n° 2011-88 de 2011, complété par les circulaires et textes d'application qui précisent certaines modalités pratiques pour les associations.",

                "Je m'appuie sur le texte intégral du décret-loi n° 2011-88, qui définit tous les aspects légaux de la vie associative en Tunisie, de la création à la dissolution, en passant par le financement et la gouvernance."
            ],
            'fallback': [
                "Je ne suis pas sûr de comprendre votre question. Je suis spécialisé dans la législation tunisienne sur les associations. Pourriez-vous reformuler ou préciser votre demande?",

                "Pardonnez-moi, mais je n'ai pas saisi complètement votre requête. Mon domaine d'expertise couvre la législation des associations en Tunisie. Comment puis-je vous aider dans ce cadre?",

                "Votre question semble en dehors de mon domaine de spécialisation, qui est le décret-loi n° 2011-88 sur les associations en Tunisie. Pourriez-vous préciser comment je peux vous aider avec ce sujet?",

                "Je ne suis pas certain de pouvoir répondre adéquatement à cette question. Je suis spécialisé dans le droit des associations en Tunisie. Y a-t-il un aspect particulier de ce domaine sur lequel je pourrais vous renseigner?",

                "Cette question ne semble pas relever directement de mon domaine d'expertise. Je peux vous aider avec la législation tunisienne des associations. Souhaitez-vous reformuler votre question dans ce contexte?"
            ],
            'continue': [
                "Je vais poursuivre sur ce sujet. ",
                "Pour continuer notre discussion, ",
                "Développons davantage ce point. ",
                "Approfondissons ce sujet. ",
                "Allons plus loin dans cette explication. ",
                "En complément de ce que je viens d'expliquer, "
            ],
            'farewell': [
                "Au revoir! N'hésitez pas à revenir si vous avez d'autres questions sur les associations en Tunisie.",
                "À bientôt! Je reste disponible pour répondre à vos futures questions sur la législation des associations.",
                "Au plaisir de vous aider à nouveau avec vos questions juridiques. Bonne journée!",
                "Merci pour notre échange. Je serai là si vous avez besoin d'autres informations sur le décret-loi n° 2011-88.",
                "À très bientôt! N'hésitez pas à me consulter pour toute question sur les associations tunisiennes.",
                "Au revoir et bonne continuation dans vos projets associatifs!"
            ]
        }

    def _initialize_follow_ups(self) -> Dict[str, List[str]]:
        """Initialize improved follow-up questions with better contextual relevance"""
        return {
            'creation': [
                "Souhaitez-vous connaître les étapes suivantes après la création de votre association?",
                "Avez-vous besoin d'informations sur les documents nécessaires pour l'enregistrement?",
                "Puis-je vous aider avec la rédaction des statuts de votre association?",
                "Avez-vous des questions sur les obligations légales après l'enregistrement?",
                "Prévoyez-vous de créer une association prochainement?",
                "Souhaitez-vous des précisions sur une étape particulière du processus de création?",
                "Envisagez-vous une association avec des partenaires internationaux?",
                "Avez-vous des questions sur la composition minimale requise pour les membres fondateurs?",
                "Le siège de votre association sera-t-il en Tunisie ou à l'étranger?",
                "Souhaitez-vous connaître les délais légaux pour chaque étape du processus de création?"
            ],
            'funding': [
                "Puis-je vous renseigner sur les obligations comptables des associations?",
                "Souhaitez-vous en savoir plus sur les différentes sources de financement autorisées?",
                "Avez-vous besoin d'informations sur la gestion des dons internationaux?",
                "Y a-t-il un aspect particulier du financement qui vous intéresse?",
                "Voulez-vous connaître les restrictions concernant certaines sources de financement?",
                "Avez-vous des questions sur la transparence financière requise pour les associations?",
                "Êtes-vous intéressé par les obligations fiscales liées aux activités lucratives?",
                "Souhaitez-vous des informations sur les subventions publiques disponibles?",
                "Avez-vous besoin de précisions sur la gestion des cotisations des membres?",
                "Voulez-vous savoir comment gérer les sponsors privés dans le cadre associatif?"
            ],
            'legal': [
                "Y a-t-il un aspect spécifique de la législation qui vous préoccupe?",
                "Avez-vous des questions sur vos obligations légales annuelles?",
                "Souhaitez-vous des informations sur les modifications statutaires?",
                "Avez-vous besoin de précisions sur les responsabilités légales des dirigeants?",
                "Y a-t-il un article particulier du décret-loi que vous souhaitez comprendre?",
                "Avez-vous des interrogations sur les relations avec les autorités publiques?",
                "Puis-je vous aider à comprendre les implications juridiques de certaines activités?",
                "Souhaitez-vous connaître les sanctions en cas de non-respect des obligations légales?",
                "Avez-vous besoin d'éclaircissements sur les obligations de reporting?",
                "Avez-vous des questions sur les droits spécifiques accordés aux associations?"
            ],
            'dissolution': [
                "Souhaitez-vous connaître les différentes causes de dissolution?",
                "Avez-vous des questions sur la procédure de liquidation?",
                "Puis-je vous informer sur la répartition des actifs après dissolution?",
                "Voulez-vous en savoir plus sur la dissolution volontaire versus judiciaire?",
                "Avez-vous besoin d'informations sur les démarches administratives de dissolution?",
                "Souhaitez-vous comprendre les responsabilités du liquidateur?",
                "Avez-vous des questions sur le devenir des documents et archives après dissolution?",
                "Vous interrogez-vous sur les conséquences fiscales d'une dissolution?",
                "Puis-je vous renseigner sur les délais légaux de la procédure de dissolution?",
                "Avez-vous besoin de conseils sur la communication avec les membres lors d'une dissolution?"
            ],
            'membership': [
                "Avez-vous des questions sur les conditions d'adhésion à une association?",
                "Souhaitez-vous connaître les droits des membres dans une association?",
                "Puis-je vous renseigner sur les différentes catégories de membres possibles?",
                "Avez-vous besoin d'informations sur l'exclusion d'un membre?",
                "Souhaitez-vous des précisions sur le rôle des membres dans les prises de décision?",
                "Avez-vous des questions sur les obligations des membres?",
                "Vous interrogez-vous sur la participation des mineurs dans une association?",
                "Souhaitez-vous connaître les règles concernant la participation des étrangers?",
                "Avez-vous besoin d'aide pour rédiger un règlement intérieur concernant les membres?",
                "Puis-je vous informer sur la gestion des conflits entre membres?"
            ],
            'general': [
                "Avez-vous d'autres questions sur les associations en Tunisie?",
                "Est-ce que je peux vous aider avec un autre aspect de la législation?",
                "Y a-t-il un sujet particulier concernant les associations qui vous intéresse?",
                "Puis-je clarifier un autre point de la loi sur les associations?",
                "Y a-t-il quelque chose de spécifique que vous aimeriez approfondir?",
                "Avez-vous besoin d'autres informations sur le décret-loi n° 2011-88?",
                "Souhaitez-vous explorer un autre aspect du droit associatif tunisien?",
                "Est-ce qu'un autre sujet lié aux associations vous préoccupe?",
                "Puis-je vous aider avec une autre question juridique concernant votre association?",
                "Y a-t-il un autre domaine du secteur associatif sur lequel vous avez des interrogations?"
            ]
        }

    def get_memory(self, conversation_id: str) -> ConversationMemory:
        """Get or initialize conversation memory"""
        if conversation_id not in self.conversation_memories:
            self.conversation_memories[conversation_id] = ConversationMemory()
        return self.conversation_memories[conversation_id]

    def get_conversation_state(self, conversation_id: str) -> Dict[str, Any]:
        """Get or initialize conversation state with enhanced tracking"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = {
                'greeting_given': False,
                'intro_given': False,
                'topic_focus': None,
                'recent_topics': [],
                'positive_sentiment': 0,
                'negative_sentiment': 0,
                'consecutive_fallbacks': 0,
                'last_response_type': None,
                'interaction_count': 0,
                'question_count': 0,
                'correction_count': 0,
                'entities_mentioned': set(),
                'user_preferences': {},
                'conversation_tone': 'neutral',  # neutral, formal, informal
                'depth_level': 'medium',  # basic, medium, detailed
                'last_activity': datetime.now(),
                'active_topics': [],
                'resolved_topics': [],
                'language': 'fr',  # default language is French
                'user_name': None,
                'formality_preference': None,  # formal, standard, informal
                'context_history': []  # Keep track of conversation context
            }
        return self.conversation_states[conversation_id]

    def update_conversation_state(self, state: Dict[str, Any], query: str, intent: str, entities: Dict[str, List[str]]):
        """Update conversation state based on the latest interaction"""
        # Update basic counters
        state['interaction_count'] += 1
        state['last_activity'] = datetime.now()

        # Track question patterns
        if '?' in query:
            state['question_count'] += 1

        # Update topic tracking
        if intent not in ['greeting', 'how_are_you', 'thank_you', 'goodbye', 'help', 'identity']:
            if intent not in state['recent_topics']:
                state['recent_topics'].append(intent)

            # Keep recent topics list capped at last 5
            if len(state['recent_topics']) > 5:
                state['recent_topics'] = state['recent_topics'][-5:]

            # Track active topics
            if intent not in state['active_topics'] and intent not in state['resolved_topics']:
                state['active_topics'].append(intent)

        # Track entities for better context
        for entity_type, entity_values in entities.items():
            for entity in entity_values:
                if entity_type == 'person' and not state['user_name'] and 'je m\'appelle' in query.lower():
                    state['user_name'] = entity
                state['entities_mentioned'].add((entity_type, entity))

        # Update context history
        context_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'intent': intent,
            'entities': entities
        }
        state['context_history'].append(context_entry)

        # Keep context history to reasonable size
        if len(state['context_history']) > 10:
            state['context_history'] = state['context_history'][-10:]

        # Detect language preference changes
        if any(word in query.lower() for word in ['english', 'anglais', 'in english', 'en anglais']):
            state['language'] = 'en'
        elif any(word in query.lower() for word in ['français', 'french', 'en français']):
            state['language'] = 'fr'

        # Detect formality preference
        if re.search(r'\b(tu|ton|tes|toi)\b', query.lower()):
            state['formality_preference'] = 'informal'
        elif re.search(r'\b(vous|votre|vos)\b', query.lower()) and len(query.split()) > 3:
            state['formality_preference'] = 'formal'

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query for better context understanding"""
        entities = {}

        # Check for each entity type using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates

        return entities

    def detect_intent(self, query: str) -> str:
        """Detect user intent with improved pattern matching and topic classification"""
        query_lower = query.lower()

        # Define intent patterns with better coverage for associations domain
        intents = {
            'creation': r'\b(créer|comment créer|fonder|établir|constituer|mettre en place|démarrer|lancer|commencer|ouvrir|formation|constitution|création)\b.*\b(association|ong|organisme|groupe)\b',
            'funding': r'\b(argent|financer|financement|budget|subvention|donation|don|ressource|revenu|cotisation|trésorerie|compte bancaire|financier)\b',
            'legal': r'\b(légal|légalité|loi|règlement|statut|juridique|droit|obligation|responsabilité|conformité|texte|décret|article)\b',
            'dissolution': r'\b(dissoudre|dissolution|fermer|fermeture|mettre fin|terminer|clôture|arrêter|cesser|liquidation|fusion)\b',
            'membership': r'\b(membre|adhésion|adhérent|rejoindre|participer|appartenir|faire partie|adhérer|s\'inscrire|inscription|cotisant)\b',
            'administration': r'\b(admin|gérer|gestion|conseil|bureau|président|secrétaire|trésorier|dirigeant|responsable|comité|directeur)\b',
            'reporting': r'\b(rapport|déclaration|signaler|rapport annuel|bilan|compte rendu|déclaration fiscale|comptabilité|transparence)\b',
            'meetings': r'\b(réunion|assemblée|générale|ordinaire|extraordinaire|conseil|séance|rencontre|ordre du jour|convocation|délibération)\b',
            'activities': r'\b(activité|projet|programme|événement|manifestation|action|initiative|organiser|atelier|conférence)\b',
            'rights': r'\b(droit|liberté|protection|garantie|prérogative|avantage|bénéfice|privilège)\b',
            'obligations': r'\b(obligation|devoir|exigence|responsabilité|nécessité|contrainte|engagement|impératif)\b',
        }

        # Check for intent matches
        for intent, pattern in intents.items():
            if re.search(pattern, query_lower):
                logger.info(f"Detected domain intent '{intent}' for query: {query}")
                return intent

        # Check for conversation pattern matches
        for pattern_type, pattern in self.patterns.items():
            if re.search(pattern, query_lower):
                logger.info(f"Detected conversation pattern '{pattern_type}' for query: {query}")
                return pattern_type

        # Default to general if no specific intent is found
        return 'general'

    def analyze_sentiment(self, query: str) -> Dict[str, float]:
        """
        Analyze sentiment of user query with more nuanced scoring
        Returns a dictionary with positive and negative sentiment scores
        """
        query_lower = query.lower()

        sentiment = {
            'positive': 0.0,
            'negative': 0.0,
            'overall': 0.0
        }

        # Count positive and negative words
        positive_count = 0
        for word in self.sentiment_words['positive']:
            if re.search(r'\b' + word + r'\b', query_lower):
                positive_count += 1

        negative_count = 0
        for word in self.sentiment_words['negative']:
            if re.search(r'\b' + word + r'\b', query_lower):
                negative_count += 1

        # Convert counts to normalized scores
        word_count = len(query_lower.split())
        if word_count > 0:
            sentiment['positive'] = min(1.0, positive_count / (word_count * 0.5))
            sentiment['negative'] = min(1.0, negative_count / (word_count * 0.5))

        # Calculate overall sentiment
        sentiment['overall'] = sentiment['positive'] - sentiment['negative']

        # Analyze punctuation for sentiment clues
        exclamation_count = query.count('!')
        question_count = query.count('?')

        sentiment['positive'] += min(0.3, exclamation_count * 0.1)  # Exclamations can indicate positive emotion
        sentiment['negative'] += min(0.2, question_count * 0.05)    # Multiple questions might indicate confusion

        # Detect negation which could flip sentiment
        if re.search(r'\b(ne|pas|non|aucun|jamais|rien)\b', query_lower):
            # Potential negation detected, this could flip the sentiment
            if sentiment['positive'] > sentiment['negative']:
                # Reduce positive sentiment if negation is present
                sentiment['positive'] *= 0.5
                sentiment['negative'] += 0.2

        # Cap final scores at 1.0
        sentiment['positive'] = min(1.0, sentiment['positive'])
        sentiment['negative'] = min(1.0, sentiment['negative'])
        sentiment['overall'] = max(-1.0, min(1.0, sentiment['overall']))

        return sentiment

    def handle_conversation(self, query: str, conversation=None) -> Tuple[Optional[str], bool]:
        """
        Handle conversational aspects of the user query with improved context awareness

        Args:
            query: The user's question
            conversation: Optional conversation object for context

        Returns:
            response: A response if conversational, None if domain-specific
            is_conversational: Boolean indicating if this was a conversational exchange
        """
        conversation_id = str(conversation.id) if conversation else 'default'
        state = self.get_conversation_state(conversation_id)
        memory = self.get_memory(conversation_id)

        # Process the query
        intent = self.detect_intent(query)
        entities = self.extract_entities(query)
        sentiment = self.analyze_sentiment(query)

        # Update state with latest interaction data
        self.update_conversation_state(state, query, intent, entities)

        query_lower = query.lower()

        # Update sentiment tracking in state
        state['positive_sentiment'] += sentiment['positive']
        state['negative_sentiment'] += sentiment['negative']

        # Extract potential user name for personalization
        name_patterns = [
            r'je m\'appelle (\w+)',
            r'mon nom est (\w+)',
            r'c\'est (\w+)',
            r'appelle[sz]?[ -]moi (\w+)'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                name = match.group(1).capitalize()
                state['user_name'] = name
                memory.remember('user_name', name, importance=8)
                logger.info(f"Extracted user name: {name}")

        # Store specific preferences if mentioned
        if re.search(r'\b(préfère|préférence|aime|voudrais|veux)\b.*\b(détail|détaillé|complet|exhaustif|approfondi)\b', query_lower):
            state['depth_level'] = 'detailed'
            memory.remember('prefers_detail', True, importance=7)

        if re.search(r'\b(préfère|préférence|aime|voudrais|veux)\b.*\b(simple|basique|court|bref|concis)\b', query_lower):
            state['depth_level'] = 'basic'
            memory.remember('prefers_simplicity', True, importance=7)

        # Check for directly conversational patterns
        for pattern_type, pattern in self.patterns.items():
            if re.search(pattern, query_lower):
                # Select a response based on pattern, with some variety based on sentiment
                responses = self.responses[pattern_type]

                # Choose response considering sentiment and previous interactions
                if sentiment['overall'] > 0.3:
                    # More enthusiastic responses for positive sentiment
                    response_index = min(len(responses) - 1, int(len(responses) * 0.7))
                    response_candidates = responses[response_index:]
                elif sentiment['overall'] < -0.3:
                    # More reassuring responses for negative sentiment
                    response_index = min(len(responses) - 1, int(len(responses) * 0.3))
                    response_candidates = responses[:response_index]
                else:
                    # Neutral responses
                    response_index = min(len(responses) - 1, int(len(responses) * 0.3))
                    response_end = min(len(responses) - 1, int(len(responses) * 0.7))
                    response_candidates = responses[response_index:response_end]

                # Add some randomness
                final_response = random.choice(response_candidates if isinstance(response_candidates, list) else [response_candidates])

                # Add name personalization if available
                if state['user_name'] and random.random() > 0.6:
                    greeting_phrases = ["", f"{state['user_name']}, ", f"Bien, {state['user_name']}, "]
                    final_response = random.choice(greeting_phrases) + final_response

                # Remember this interaction in memory
                memory.remember(f"conversational_{pattern_type}", query, importance=3,
                               expiry=datetime.now() + timedelta(hours=1))

                # Update state
                state['last_response_type'] = pattern_type
                state['consecutive_fallbacks'] = 0

                logger.info(f"Matched conversation pattern '{pattern_type}', returning conversational response")
                return final_response, True

        # If we've reached here, it's not a direct conversational pattern
        return None, False

    def get_greeting(self, state=None) -> str:
        """Returns a more personalized greeting appropriate for the time of day and user history"""
        hour = datetime.now().hour

        # Personalize based on user's name if available
        name_phrase = f", {state['user_name']}" if state and state.get('user_name') else ""

        # Adjust greeting based on time of day
        if 5 <= hour < 12:
            greetings = [
                f"Bonjour{name_phrase}! Je suis votre assistant pour la législation tunisienne sur les associations.",
                f"Bonjour et bienvenue{name_phrase}! Comment puis-je vous aider aujourd'hui?",
                f"Bonjour{name_phrase}! J'espère que votre matinée se passe bien. Comment puis-je vous assister?"
            ]
        elif 12 <= hour < 18:
            greetings = [
                f"Bon après-midi{name_phrase}! Je suis votre assistant pour la législation tunisienne sur les associations.",
                f"Bonjour{name_phrase}! J'espère que vous passez une bonne journée. En quoi puis-je vous aider?",
                f"Bonjour{name_phrase}! Comment puis-je vous assister cet après-midi?"
            ]
        else:
            greetings = [
                f"Bonsoir{name_phrase}! Je suis votre assistant pour la législation tunisienne sur les associations.",
                f"Bonsoir{name_phrase}! Comment puis-je vous aider en cette soirée?",
                f"Bonsoir{name_phrase}! J'espère que votre journée s'est bien passée. En quoi puis-je vous être utile?"
            ]

        # Add formality variation based on preference if available
        if state and state.get('formality_preference') == 'informal':
            informal_greetings = [
                f"Salut{name_phrase}! Je suis ton assistant pour les associations tunisiennes. Comment je peux t'aider?",
                f"Coucou{name_phrase}! Comment vas-tu? Je suis là pour répondre à tes questions sur les associations.",
                f"Hey{name_phrase}! Content de te retrouver. Que puis-je faire pour toi aujourd'hui?"
            ]
            greetings.extend(informal_greetings)

        return random.choice(greetings)

    def get_tailored_followup(self, query: str, intent: str, conversation=None) -> str:
        """Generate a more natural follow-up based on conversation context and inferred interests"""
        conversation_id = str(conversation.id) if conversation else 'default'
        state = self.get_conversation_state(conversation_id)
        memory = self.get_memory(conversation_id)

        # Update recent topics
        if intent not in ['greeting', 'how_are_you', 'thank_you', 'goodbye']:
            state['recent_topics'].append(intent)

        # Limit recent topics list to last 5
        if len(state['recent_topics']) > 5:
            state['recent_topics'] = state['recent_topics'][-5:]

        # Get topic counts for personalization
        topic_counts = {}
        for topic in state['recent_topics']:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Find most frequent topic
        most_frequent_topic = max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else 'general'

        # If we've discussed this topic multiple times, go deeper with followups
        deeper_followup = False
        if most_frequent_topic in topic_counts and topic_counts[most_frequent_topic] > 1:
            deeper_followup = True

        # Look for related topics we could suggest
        related_topics = []
        if intent in self.topic_transitions:
            related_topics = self.topic_transitions[intent]

        # Select appropriate followups based on context
        if intent in self.follow_ups and deeper_followup:
            # Use more specific followups for repeated topics - choose from the second half of the list
            specific_followups = self.follow_ups.get(intent, self.follow_ups['general'])
            mid_point = len(specific_followups) // 2
            return random.choice(specific_followups[mid_point:])

        elif intent in self.follow_ups:
            # Use general followups for first encounter with topic - choose from the first half
            general_followups = self.follow_ups.get(intent, self.follow_ups['general'])
            mid_point = len(general_followups) // 2
            return random.choice(general_followups[:mid_point])

        elif related_topics:
            # Suggest a related topic if we haven't discussed any deeply
            related_topic = random.choice(related_topics)

            related_prompts = [
                f"Souhaitez-vous également des informations sur les aspects de {related_topic} liés à votre question?",
                f"Puis-je vous renseigner sur {related_topic}, qui est souvent lié à ce sujet?",
                f"Avez-vous aussi des questions concernant {related_topic}?",
                f"Pour compléter, souhaiteriez-vous des informations sur {related_topic}?"
            ]
            return random.choice(related_prompts)

        else:
            # Fallback to general followups
            return random.choice(self.follow_ups['general'])

    def generate_response_transition(self, state, intent):
        """Generate natural transitions between topics in a conversation"""
        # If this is a new topic after discussing something else
        if state['recent_topics'] and state['recent_topics'][-1] != intent and len(state['recent_topics']) > 1:
            previous_topic = state['recent_topics'][-2]

            transitions = [
                f"Passons maintenant au sujet de {intent}. ",
                f"En ce qui concerne votre question sur {intent}, ",
                f"Pour répondre à votre interrogation sur {intent}, ",
                f"Abordons maintenant la question de {intent}. ",
                f"Sur le sujet de {intent} que vous venez d'évoquer, "
            ]

            return random.choice(transitions)

        return ""

    def enhance_response(self, response: str, query: str, intent: str, conversation=None) -> str:
        """
        Enhance a response with conversational elements, personalization, and context awareness

        Args:
            response: The base response to enhance
            query: The original query
            intent: The detected intent
            conversation: Optional conversation object

        Returns:
            An enhanced, more natural response
        """
        conversation_id = str(conversation.id) if conversation else 'default'
        state = self.get_conversation_state(conversation_id)
        memory = self.get_memory(conversation_id)

        is_first_interaction = state['interaction_count'] <= 1
        sentiment = self.analyze_sentiment(query)

        # Start with the original response
        enhanced_response = response

        # Add a greeting for the first interaction
        if is_first_interaction:
            greeting = self.get_greeting(state)
            enhanced_response = f"{greeting}\n\n{enhanced_response}"

        # Add natural transitions for topic changes
        else:
            transition = self.generate_response_transition(state, intent)
            if transition and not enhanced_response.startswith(transition):
                enhanced_response = f"{transition}{enhanced_response[0].lower()}{enhanced_response[1:]}"

        # For non-first interactions, occasionally add personalization
        if not is_first_interaction and state.get('user_name') and random.random() > 0.7:
            # Find a good place to insert the name in the response
            sentences = re.split(r'(?<=[.!?])\s+', enhanced_response)
            if len(sentences) > 1:
                # Add name to second sentence for more natural flow
                second_sentence = sentences[1]
                if second_sentence.startswith(("Je", "J'", "Vous", "Votre")):
                    sentences[1] = f"{state['user_name']}, {second_sentence[0].lower()}{second_sentence[1:]}"
                enhanced_response = " ".join(sentences)

        # Adjust formality based on user preference
        if state.get('formality_preference') == 'informal' and random.random() > 0.5:
            # Make response more informal by replacing formal pronouns
            enhanced_response = enhanced_response.replace("Vous pouvez", "Tu peux")
            enhanced_response = enhanced_response.replace("vous avez", "tu as")
            enhanced_response = enhanced_response.replace("votre", "ton")
            enhanced_response = enhanced_response.replace("vos", "tes")

        # Detect if the response doesn't contain a clear answer
        uncertainty_phrases = ["je ne peux pas répondre",
                              "je n'ai pas d'information",
                              "je ne suis pas en mesure",
                              "je ne suis pas sûr",
                              "je n'ai pas trouvé"]

        if any(phrase in enhanced_response.lower() for phrase in uncertainty_phrases):
            # Add apology and recovery for failed responses
            if random.random() > 0.5:
                enhanced_response += "\n\nJe m'excuse de ne pas pouvoir mieux répondre à votre question. Puis-je vous aider avec un autre aspect de la législation des associations en Tunisie?"
            else:
                enhanced_response += "\n\nPuis-je vous aider avec un autre sujet concernant les associations en Tunisie?"

            # Increment fallback counter
            state['consecutive_fallbacks'] += 1
        else:
            # Reset fallback counter on successful response
            state['consecutive_fallbacks'] = 0

        # Add variety through different response structures based on length and interaction count
        response_words = len(response.split())
        interaction_count = state['interaction_count']

        if response_words < 40:
            # For shorter responses, add a tailored follow-up question
            followup = self.get_tailored_followup(query, intent, conversation)
            enhanced_response += f"\n\n{followup}"

        elif response_words < 80:
            # For medium responses, sometimes add a follow-up
            if random.random() > 0.4:  # 60% chance
                followup = self.get_tailored_followup(query, intent, conversation)
                enhanced_response += f"\n\n{followup}"

        else:
            # For longer, detailed responses, add a simple closing with lower probability
            if random.random() > 0.7:  # 30% chance
                closings = [
                    "J'espère que cette explication vous a été utile. N'hésitez pas à me poser d'autres questions.",
                    "Ces informations répondent-elles à votre question? Je reste à votre disposition.",
                    "Est-ce que ces détails vous aident à comprendre la situation? Avez-vous d'autres questions?"
                ]
                enhanced_response += f"\n\n{random.choice(closings)}"

        # If negative sentiment is high or multiple fallbacks, add reassurance
        if sentiment['negative'] > 0.6 or state['consecutive_fallbacks'] >= 2:
            reassurances = [
                "Je comprends que ce sujet peut être complexe. N'hésitez pas à me demander des clarifications si nécessaire.",
                "Si ma réponse ne correspond pas à vos attentes, n'hésitez pas à reformuler votre question.",
                "Je suis là pour vous aider à naviguer dans ces questions juridiques. N'hésitez pas à me préciser vos besoins."
            ]
            enhanced_response += f"\n\n{random.choice(reassurances)}"

        # If it's been several interactions, occasionally add a "checking in" question
        if interaction_count > 3 and random.random() > 0.8:
            check_ins = [
                "Ces informations vous sont-elles utiles?",
                "Est-ce que je réponds bien à vos attentes?",
                "La conversation répond-elle à vos besoins?",
                "Suis-je suffisamment clair dans mes explications?"
            ]
            enhanced_response += f"\n\n{random.choice(check_ins)}"

        return enhanced_response


# Instantiate the conversation manager globally to maintain state
conversation_manager = ConversationManager()