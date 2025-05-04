import os
import logging
import traceback
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set
from django.conf import settings
from .models import Conversation, Message, DocumentChunk, Document
from .utils import find_relevant_chunks, analyze_document_structure, preprocess_query
from .conversation_handlers import conversation_manager

logger = logging.getLogger(__name__)


class ChatbotService:
    """Enhanced service for handling chatbot interactions with multiple LLM options and advanced reasoning"""

    def __init__(self):
        # Initialize with no models loaded
        self.local_llm = None
        self.local_llm_available = False
        self.local_model_loaded = False

        # API LLM settings
        self.use_api_llm = getattr(settings, 'USE_API_LLM', True)  # Default to API for better quality
        self.api_provider = getattr(settings, 'API_LLM_PROVIDER', 'anthropic')  # or 'openai', etc.

        # Language Detection
        self.supported_languages = ['fr', 'en', 'ar']  # French, English, Arabic
        self.default_language = 'fr'

        # Performance metrics
        self.response_times = []

        # Cache frequently used data
        self.document_cache = {}
        self.article_cache = {}

        # Initialize article structure if available
        self._initialize_document_structure()

    def _initialize_document_structure(self):
        """Initialize document structure data for faster article lookup"""
        try:
            documents = Document.objects.all()

            for document in documents:
                # Analyze document structure to extract articles
                structure = analyze_document_structure(document.id)

                if structure and 'articles' in structure:
                    # Cache the articles for this document
                    self.article_cache[document.id] = structure['articles']
                    logger.info(f"Cached {len(structure['articles'])} articles for document {document.id}")
        except Exception as e:
            logger.error(f"Error initializing document structure: {e}")

    def _load_local_model_if_needed(self):
        """Lazy-load the local model only when needed with better error handling"""
        if self.local_model_loaded:
            return self.local_llm_available

        try:
            from llama_cpp import Llama
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'mistral-7b-instruct-v0.1.Q4_K_M.gguf')

            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path} with optimized settings")

                try:
                    # Use more memory-efficient settings
                    self.local_llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,  # Increased context window
                        n_threads=4,  # Balancing threads for better performance
                        n_batch=8,  # Smaller batch size
                        use_mlock=False  # Don't lock memory
                    )
                    self.local_llm_available = True
                    logger.info(f"LLM loaded successfully with memory-optimized settings")
                except Exception as e:
                    logger.error(f"Error loading LLM with initial settings: {e}")
                    logger.info("Trying with minimal settings...")

                    # Try with absolute minimal settings
                    try:
                        self.local_llm = Llama(
                            model_path=model_path,
                            n_ctx=512,
                            n_threads=1,
                            n_batch=1
                        )
                        self.local_llm_available = True
                        logger.info("LLM loaded with minimal settings")
                    except Exception as e:
                        logger.error(f"Error loading LLM with minimal settings: {e}")
                        self.local_llm_available = False
            else:
                logger.warning(f"LLM model file not found at {model_path}")
                self.local_llm_available = False

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.local_llm_available = False

        self.local_model_loaded = True
        return self.local_llm_available

    def _create_system_prompt(self, relevant_chunks, conversation_history=None, language='fr'):
        """
        Create a more sophisticated system prompt with context from relevant document chunks
        and conversation history for better coherence and contextual awareness
        """
        # Get content from chunks or use empty string if no chunks
        chunks_text = "\n\n".join([chunk.content for chunk in relevant_chunks]) if relevant_chunks else ""

        # Extract articles mentioned in conversation
        mentioned_articles = set()
        if conversation_history:
            for msg in conversation_history:
                # Look for mentions of specific articles
                article_mentions = re.findall(r'article\s+(\d+)', msg.get('content', '').lower())
                mentioned_articles.update(article_mentions)

        # Add specific article text if mentioned and available
        article_context = ""
        if mentioned_articles and self.article_cache:
            # For each document in cache
            for doc_id, articles in self.article_cache.items():
                # For each mentioned article number
                for article_num in mentioned_articles:
                    if article_num in articles:
                        article_context += f"\nArticle {article_num}:\n{articles[article_num]}\n"

        # Add article context if available
        if article_context:
            chunks_text += "\n\nARTICLES SPÉCIFIQUEMENT MENTIONNÉS:\n" + article_context

        # Select language-appropriate system prompt
        if language == 'en':
            return f"""You are a friendly and helpful AI assistant specializing in Tunisian association law, based on Decree-Law No. 2011-88 of September 24, 2011.

CONTEXT:
{chunks_text}

INSTRUCTIONS:
1. Always answer in English unless explicitly requested otherwise.
2. Be friendly, patient and helpful in your responses. Use natural and conversational language.
3. If the answer is in the provided context, use this information and cite specific articles.
4. Vary your response style to avoid monotony. Adapt your level of detail to the complexity of the question.
5. If the question is off-topic, politely explain that you specialize in Tunisian association law.
6. If the answer is not in the excerpts or if you're not sure, clearly state so without inventing information.
7. You can answer general questions and participate in normal conversations in addition to specific questions about associations.
8. Be precise, professional and concise in your responses.
9. Cite relevant articles from the decree-law when appropriate.
10. If the user makes spelling or grammar mistakes, still understand their question and respond appropriately."""
        elif language == 'ar':
            return f"""أنت مساعد ذكاء اصطناعي ودود ومفيد متخصص في قانون الجمعيات التونسي، استنادًا إلى المرسوم بقانون رقم 2011-88 المؤرخ في 24 سبتمبر 2011.

السياق:
{chunks_text}

تعليمات:
1. أجب دائمًا باللغة العربية ما لم يُطلب منك صراحةً خلاف ذلك.
2. كن ودودًا وصبورًا ومفيدًا في ردودك. استخدم لغة طبيعية ومحادثة.
3. إذا كانت الإجابة موجودة في السياق المقدم، استخدم هذه المعلومات واستشهد بمواد محددة.
4. نوع أسلوب ردك لتجنب الرتابة. قم بتكييف مستوى التفاصيل مع تعقيد السؤال.
5. إذا كان السؤال خارج الموضوع، اشرح بأدب أنك متخصص في قانون الجمعيات التونسي.
6. إذا لم تكن الإجابة في المقتطفات أو إذا لم تكن متأكدًا، فعليك أن تذكر ذلك بوضوح دون اختلاق معلومات.
7. يمكنك الإجابة على الأسئلة العامة والمشاركة في المحادثات العادية بالإضافة إلى الأسئلة المحددة حول الجمعيات.
8. كن دقيقًا ومهنيًا وموجزًا في ردودك.
9. استشهد بالمواد ذات الصلة من المرسوم بقانون عند الاقتضاء.
10. إذا أخطأ المستخدم في التهجئة أو قواعد اللغة، فلا يزال بإمكانك فهم سؤاله والرد بشكل مناسب."""
        else:  # Default to French
            return f"""Tu es un assistant IA convivial et serviable spécialisé dans la législation tunisienne sur les associations, basé sur le Décret-loi n° 2011-88 du 24 septembre 2011.

CONTEXTE:
{chunks_text}

INSTRUCTIONS:
1. Réponds toujours en français sauf si explicitement demandé autrement.
2. Sois amical, patient et serviable dans tes réponses. Utilise un langage naturel et conversationnel.
3. Si la réponse se trouve dans le contexte fourni, utilise cette information et cite les articles spécifiques.
4. Varie ton style de réponse pour éviter la monotonie. Adapte ton niveau de détail à la complexité de la question.
5. Si la question est hors sujet, explique poliment que tu es spécialisé dans la législation tunisienne sur les associations.
6. Si la réponse n'est pas dans les extraits ou si tu n'es pas sûr, dis-le clairement sans inventer d'informations.
7. Tu peux répondre à des questions générales et participer à des conversations normales en plus des questions spécifiques sur les associations.
8. Sois précis, professionnel et concis dans tes réponses.
9. Cite les articles pertinents du décret-loi quand c'est approprié.
10. Si l'utilisateur fait des fautes d'orthographe ou de grammaire, comprends quand même sa question et réponds de manière appropriée."""

    def _get_conversation_history(self, conversation, max_messages=10):
        """
        Get recent conversation history with dynamic context window
        """
        # Get all messages for this conversation, ordered by time
        messages = conversation.messages.order_by('created_at')

        # Get total count
        message_count = len(messages)

        # If very few messages, include all of them
        if message_count <= max_messages:
            selected_messages = messages
        else:
            # For longer conversations, use a sliding window approach
            # Include the first message for context, then the most recent messages
            selected_messages = [messages[0]]  # Always include the first message

            # Add the most recent messages to fill our context window
            recent_count = min(max_messages - 1, message_count - 1)
            selected_messages.extend(messages[message_count - recent_count:])

        # Convert to message format for LLM consumption
        history = []
        for msg in selected_messages:
            if msg.role != 'system':  # Skip system messages in the history
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return history

    def detect_language(self, text):
        """Detect the language of the input text"""
        # Simple language detection based on common words
        text_lower = text.lower()

        # French indicators
        french_words = ['bonjour', 'salut', 'merci', 'comment', 'je', 'tu', 'vous', 'nous', 'ils', 'le', 'la', 'les',
                        'et', 'pour']
        french_count = sum(1 for word in french_words if f" {word} " in f" {text_lower} ")

        # English indicators
        english_words = ['hello', 'hi', 'thanks', 'thank', 'how', 'are', 'you', 'is', 'the', 'and', 'for', 'to', 'we',
                         'they']
        english_count = sum(1 for word in english_words if f" {word} " in f" {text_lower} ")

        # Arabic indicators
        arabic_words = ['سلام', 'مرحبا', 'شكرا', 'كيف', 'أنا', 'أنت', 'هو', 'هي', 'نحن', 'ال', 'و', 'في', 'من', 'على']
        arabic_count = sum(1 for word in arabic_words if word in text_lower)

        # Determine language based on word counts
        counts = {'fr': french_count, 'en': english_count, 'ar': arabic_count}
        detected = max(counts, key=counts.get)

        # Only return detected language if we have enough confidence
        if counts[detected] >= 2:
            return detected

        # Default to French if unsure
        return self.default_language

    def _find_specific_articles(self, query):
        """
        Find specific articles mentioned in the query
        """
        article_mentions = re.findall(r'article\s+(\d+)', query.lower())

        specific_articles = []
        if article_mentions and self.article_cache:
            for doc_id, articles in self.article_cache.items():
                for article_num in article_mentions:
                    if article_num in articles:
                        specific_articles.append({
                            'article_num': article_num,
                            'content': articles[article_num],
                            'document_id': doc_id
                        })

        return specific_articles

    def _generate_response_with_api_llm(self, query, relevant_chunks, conversation_history=None, language='fr'):
        """Generate a response using an API-based LLM with enhanced prompting"""
        try:
            # Create system prompt from relevant chunks
            system_prompt = self._create_system_prompt(relevant_chunks, conversation_history, language)

            # Find any specific articles mentioned
            specific_articles = self._find_specific_articles(query)

            # Add any specifically mentioned articles that weren't in the chunks
            if specific_articles:
                article_context = "\n\nARTICLES SPÉCIFIQUEMENT DEMANDÉS:\n"
                for article in specific_articles:
                    article_context += f"\nArticle {article['article_num']}:\n{article['content']}\n"
                system_prompt += article_context

            # Use appropriate API based on settings
            if self.api_provider == 'anthropic':
                return self._generate_with_anthropic(query, system_prompt, conversation_history, language)
            elif self.api_provider == 'openai':
                return self._generate_with_openai(query, system_prompt, conversation_history, language)
            else:
                logger.error(f"Unknown API provider: {self.api_provider}")
                return self._generate_fallback_response(query, language)

        except Exception as e:
            logger.error(f"Error generating response with API LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_fallback_response(query, language)

    def _generate_with_anthropic(self, query, system_prompt, conversation_history=None, language='fr'):
        """Generate response using Anthropic Claude API with enhanced prompting"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            messages = [
                {"role": "system", "content": system_prompt},
            ]

            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the current query
            messages.append({"role": "user", "content": query})

            logger.info(f"Generating response with Anthropic API for query: {query}")

            # Select appropriate Claude model based on query complexity
            model = "claude-3-haiku-20240307"  # Faster model for simple queries
            if len(query.split()) > 20 or len(system_prompt) > 1000:
                model = "claude-3-sonnet-20240229"  # More capable model for complex queries

            # If specifically asking for articles, use Opus for better accuracy
            if re.search(r'article\s+\d+', query.lower()):
                model = "claude-3-opus-20240229"  # Most capable model for specific legal questions

            logger.info(f"Selected Claude model: {model}")

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.7,
                messages=messages
            )

            response_text = response.content[0].text
            logger.info(f"Generated response with Anthropic API: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"Error with Anthropic API: {e}")
            raise

    def _generate_with_openai(self, query, system_prompt, conversation_history=None, language='fr'):
        """Generate response using OpenAI API with enhanced model selection"""
        try:
            import openai
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

            messages = [
                {"role": "system", "content": system_prompt},
            ]

            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the current query
            messages.append({"role": "user", "content": query})

            logger.info(f"Generating response with OpenAI API for query: {query}")

            # Select appropriate model based on query complexity
            model = "gpt-3.5-turbo"  # Default for simple queries

            # Use GPT-4 for complex queries or when articles are specifically mentioned
            if len(query.split()) > 30 or re.search(r'article\s+\d+', query.lower()):
                model = "gpt-4"

            logger.info(f"Selected OpenAI model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            logger.info(f"Generated response with OpenAI API: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            raise

    def _generate_response_with_local_llm(self, query, relevant_chunks, conversation_history=None, language='fr'):
        """Generate a response using the local LLM with better error handling"""
        if not self._load_local_model_if_needed():
            logger.warning("Local LLM not available, using fallback response")
            return self._generate_fallback_response(query, language)

        try:
            # Create a system prompt
            system_prompt = self._create_system_prompt(relevant_chunks, conversation_history, language)

            # Extract specific articles if mentioned
            specific_articles = self._find_specific_articles(query)

            # For local models, we'll use a simpler approach due to context constraints
            chunks_text = "\n\n".join([chunk.content for chunk in relevant_chunks]) if relevant_chunks else ""

            # Add specific articles content if available
            if specific_articles:
                article_text = "\n".join(
                    [f"Article {a['article_num']}: {a['content'][:200]}..." for a in specific_articles])
                chunks_text += f"\n\nARTICLES MENTIONNÉS:\n{article_text}"

            # Format based on language
            if language == 'en':
                simple_prompt = f"""Based on excerpts from Decree-Law No. 2011-88 on associations in Tunisia:

{chunks_text}

Question: {query}

Answer:"""
            elif language == 'ar':
                simple_prompt = f"""استنادًا إلى مقتطفات من المرسوم بقانون رقم 2011-88 بشأن الجمعيات في تونس:

{chunks_text}

سؤال: {query}

إجابة:"""
            else:  # Default to French
                simple_prompt = f"""Sur base des extraits du décret-loi n° 2011-88 du 24 septembre 2011 portant organisation des associations en Tunisie:

{chunks_text}

Question: {query}

Réponse:"""

            logger.info(f"Generating response for query: {query}")
            logger.info(f"Using {len(relevant_chunks)} relevant chunks")

            # Try simple completion first
            try:
                logger.info("Attempting simple text completion")
                output = self.local_llm(
                    simple_prompt,
                    max_tokens=512,  # Increased from 256
                    temperature=0.7,
                    stop=["Question:", "\n\n\n"]
                )
                response_text = output['choices'][0]['text'].strip()
                logger.info(f"Generated response with simple completion: {response_text[:100]}...")
                return response_text
            except Exception as e:
                logger.error(f"Error with simple completion: {e}")

                # Fall back to chat completion if simple completion fails
                logger.info("Falling back to chat completion")

                # Build messages for chat completion
                messages = [{"role": "system", "content": system_prompt}]

                # Keep history minimal for local model
                if conversation_history:
                    # Just add the last message if available
                    if len(conversation_history) > 0:
                        messages.append(conversation_history[-1])

                # Add the current query
                messages.append({"role": "user", "content": query})

                # Try chat completion
                response = self.local_llm.create_chat_completion(
                    messages=messages,
                    max_tokens=384,  # Still limited
                    temperature=0.7
                )

                generated_text = response["choices"][0]["message"]["content"]
                logger.info(f"Generated response with chat completion: {generated_text[:100]}...")
                return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response with local LLM: {str(e)}")
            logger.error(traceback.format_exc())

            # Try with a hardcoded response based on the query
            return self._generate_fallback_response(query, language)

    def _generate_fallback_response(self, query, language='fr'):
        """
        Generate a smarter rule-based response when LLM is not available
        """
        # Simple pattern matching for common questions
        query_lower = query.lower()

        # Check if it's a conversational query first
        conversation_response, is_conversational = conversation_manager.handle_conversation(query)
        if is_conversational:
            return conversation_response

        # Choose appropriate language for fallback responses
        if language == 'en':
            # English fallback responses
            if any(word in query_lower for word in ["create", "establish", "found"]) and "association" in query_lower:
                return ("To create an association in Tunisia, you must follow the declaration regime according to "
                        "Article 10 of Decree-Law No. 2011-88. Send a registered letter to the Secretary General "
                        "of the Government with the required documents: declaration, copies of founders' ID cards, "
                        "and two copies of the signed statutes.")

            elif "statute" in query_lower or "statutes" in query_lower:
                return ("The statutes of an association must contain: the official name in Arabic and in a foreign "
                        "language if applicable, the address of the headquarters, the objectives, membership conditions, "
                        "the organizational chart, decision-making procedures, and the amount of the membership fee if it exists.")

            elif any(word in query_lower for word in ["finance", "funding", "resources", "money", "budget"]):
                return (
                    "According to Article 34, an association's resources may include: member contributions, "
                    "public aid, donations and legacies of national or foreign origin, and income resulting from "
                    "its property, activities, and projects.")

            elif "dissolution" in query_lower:
                return (
                    "The dissolution of an association can be voluntary (by decision of its members in accordance with the "
                    "statutes) or judicial (by court judgment). In case of dissolution, a liquidator must be designated.")

            elif any(word in query_lower for word in ["membership", "member", "join"]):
                return (
                    "According to Article 17 of Decree-Law No. 2011-88, a member of an association must be of Tunisian "
                    "nationality or a resident in Tunisia, be at least 13 years old, accept in writing the statutes of the association, and pay "
                    "the membership fee if required. Founders and leaders cannot hold responsibilities in "
                    "political parties (Article 9).")

            else:
                # Generic response
                return ("I am unable to answer this specific question at the moment. "
                        "I can provide information on creating associations, statutes, funding, "
                        "dissolution, or membership conditions according to Decree-Law No. 2011-88. "
                        "How can I help you with these topics?")

        elif language == 'ar':
            # Arabic fallback responses
            if any(word in query_lower for word in ["إنشاء", "تأسيس", "تكوين"]) and (
                    "جمعية" in query_lower or "منظمة" in query_lower):
                return (
                    "لإنشاء جمعية في تونس، يجب عليك اتباع نظام التصريح وفقًا للمادة 10 من المرسوم بقانون رقم 2011-88. "
                    "أرسل رسالة مسجلة إلى الكاتب العام للحكومة مع المستندات المطلوبة: التصريح، ونسخ من بطاقات هوية المؤسسين، "
                    "ونسختين من النظام الأساسي الموقع.")

            elif "نظام" in query_lower or "قانون" in query_lower:
                return (
                    "يجب أن يحتوي النظام الأساسي للجمعية على: الاسم الرسمي باللغة العربية وبلغة أجنبية إذا كان ذلك مناسبًا، "
                    "وعنوان المقر الرئيسي، والأهداف، وشروط العضوية، والهيكل التنظيمي، وطرق اتخاذ القرار، ومبلغ الاشتراك إذا وجد.")

            elif any(word in query_lower for word in ["تمويل", "مال", "موارد", "ميزانية"]):
                return (
                    "وفقًا للمادة 34، يمكن أن تتكون موارد الجمعية من: اشتراكات أعضائها، "
                    "والمساعدات العامة، والتبرعات والهبات والوصايا ذات الأصل الوطني أو الأجنبي، والعائدات الناتجة عن "
                    "ممتلكاتها وأنشطتها ومشاريعها.")

            elif "حل" in query_lower:
                return (
                    "يمكن أن يكون حل الجمعية طوعيًا (بقرار من أعضائها وفقًا للنظام الأساسي) "
                    "أو قضائيًا (بحكم من المحكمة). في حالة الحل، يجب تعيين مصفٍ.")

            elif any(word in query_lower for word in ["عضوية", "عضو", "انضمام"]):
                return (
                    "وفقًا للمادة 17 من المرسوم بقانون رقم 2011-88، يجب أن يكون عضو الجمعية من الجنسية التونسية "
                    "أو مقيمًا في تونس، وألا يقل عمره عن 13 عامًا، وأن يقبل كتابيًا النظام الأساسي للجمعية، وأن يدفع "
                    "الاشتراك المطلوب. لا يمكن للمؤسسين والقادة تولي مسؤوليات في "
                    "الأحزاب السياسية (المادة 9).")

            else:
                # Generic response
                return ("لا يمكنني الإجابة على هذا السؤال المحدد في الوقت الحالي. "
                        "يمكنني تقديم معلومات حول إنشاء الجمعيات، والنظام الأساسي، والتمويل، "
                        "والحل، أو شروط العضوية وفقًا للمرسوم بقانون رقم 2011-88. "
                        "كيف يمكنني مساعدتك في هذه المواضيع؟")

        else:
            # French fallback responses (default)
            if any(word in query_lower for word in ["créer", "constituer", "fonder"]) and "association" in query_lower:
                return ("Pour créer une association en Tunisie, vous devez suivre le régime de déclaration selon "
                        "l'article 10 du décret-loi n° 2011-88. Envoyez une lettre recommandée au secrétaire général "
                        "du gouvernement avec les documents requis: déclaration, copies des cartes d'identité des "
                        "fondateurs, et deux exemplaires des statuts signés.")

            elif "statuts" in query_lower:
                return (
                    "Les statuts d'une association doivent contenir: la dénomination officielle en arabe et en langue "
                    "étrangère le cas échéant, l'adresse du siège, les objectifs, les conditions d'adhésion, "
                    "l'organigramme, les modalités de prise de décision et le montant de la cotisation s'il existe.")

            elif any(word in query_lower for word in ["finance", "financement", "ressources", "argent", "budget"]):
                return (
                    "Selon l'article 34, les ressources d'une association peuvent comprendre: les cotisations des membres, "
                    "les aides publiques, les dons et legs d'origine nationale ou étrangère, et les recettes résultant de "
                    "ses biens, activités et projets.")

            elif "dissolution" in query_lower:
                return (
                    "La dissolution d'une association peut être volontaire (par décision de ses membres conformément aux "
                    "statuts) ou judiciaire (par jugement du tribunal). En cas de dissolution, un liquidateur doit être désigné.")

            elif any(word in query_lower for word in ["adhésion", "membre", "adhérer"]):
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
        try:
            # Start timing for performance monitoring
            start_time = time.time()

            # Preprocess the query to handle typos and language issues
            processed_query = preprocess_query(query)
            logger.info(f"Preprocessed query: {query} -> {processed_query}")

            # Detect language
            detected_language = self.detect_language(processed_query)
            logger.info(f"Detected language: {detected_language}")

            # Check if it's a purely conversational query first
            conversation_response, is_conversational = conversation_manager.handle_conversation(
                processed_query, conversation
            )

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

                # Log performance metrics
                processing_time = time.time() - start_time
                self.response_times.append(processing_time)
                logger.info(f"Processed conversational query in {processing_time:.2f}s")

                return {
                    "message_id": assistant_message.id,
                    "content": conversation_response,
                    "relevant_documents": [],
                    "language": detected_language
                }

            # For non-conversational queries, proceed with document search
            # Find relevant document chunks with more context
            intent = conversation_manager.detect_intent(processed_query)
            relevant_chunks = find_relevant_chunks(processed_query)

            # Find specific articles if mentioned in the query
            specific_articles = self._find_specific_articles(processed_query)
            if specific_articles:
                logger.info(f"Found specific articles mentioned: {[a['article_num'] for a in specific_articles]}")

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

            # Choose LLM method based on settings and query complexity
            if self.use_api_llm:
                response_text = self._generate_response_with_api_llm(
                    processed_query, relevant_chunks, conversation_history, detected_language
                )
            else:
                response_text = self._generate_response_with_local_llm(
                    processed_query, relevant_chunks, conversation_history, detected_language
                )

            # Enhance the response with conversational elements
            enhanced_response = conversation_manager.enhance_response(
                response_text, processed_query, intent, conversation
            )

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

            # Log performance metrics
            processing_time = time.time() - start_time
            self.response_times.append(processing_time)
            avg_time = sum(self.response_times) / len(self.response_times)
            logger.info(f"Processed query in {processing_time:.2f}s (avg: {avg_time:.2f}s)")

            return {
                "message_id": assistant_message.id,
                "content": enhanced_response,
                "relevant_documents": [
                    {
                        "title": chunk.document.title,
                        "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    }
                    for chunk in relevant_chunks
                ],
                "language": detected_language
            }

        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

            # Determine appropriate language for error message
            try:
                error_language = self.detect_language(query)
            except:
                error_language = 'fr'  # Default to French if detection fails

            # Return graceful error response in the appropriate language
            if error_language == 'en':
                error_response = "I'm sorry, I encountered a technical difficulty. Could you rephrase your question or try again later?"
            elif error_language == 'ar':
                error_response = "آسف، لقد واجهت صعوبة فنية. هل يمكنك إعادة صياغة سؤالك أو المحاولة مرة أخرى لاحقًا؟"
            else:
                error_response = "Je suis désolé, j'ai rencontré une difficulté technique. Pourriez-vous reformuler votre question ou réessayer plus tard?"

            # Try to save error message
            try:
                # Save system message with error info
                Message.objects.create(
                    conversation=conversation,
                    role='system',
                    content=f"ERROR: {str(e)}\nQuery: {query}"
                )

                # Save assistant message with error response
                assistant_message = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=error_response
                )

                message_id = assistant_message.id
            except Exception:
                message_id = None

            return {
                "message_id": message_id,
                "content": error_response,
                "relevant_documents": [],
                "language": error_language
            }

    def process_message_streaming(self, conversation, query):
        """
        Process a message with streaming response
        Yields chunks of the response as they're generated
        """
        try:
            # Initial response data
            response_data = {
                "event": "start",
                "message_id": None,
                "content": "",
                "relevant_documents": []
            }

            # Start timing
            start_time = time.time()

            # Preprocess query and detect language
            processed_query = preprocess_query(query)
            detected_language = self.detect_language(processed_query)
            response_data["language"] = detected_language

            # Check if it's conversational
            conversation_response, is_conversational = conversation_manager.handle_conversation(processed_query,
                                                                                                conversation)

            # Save user message
            user_message = Message.objects.create(
                conversation=conversation,
                role='user',
                content=query
            )

            # For conversational queries, return the response immediately
            if is_conversational:
                # Save assistant message
                assistant_message = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=conversation_response
                )

                # Send the full response
                response_data["event"] = "message"
                response_data["message_id"] = assistant_message.id
                response_data["content"] = conversation_response
                yield f"data: {json.dumps(response_data)}\n\n"

                # Send completion event
                response_data["event"] = "done"
                yield f"data: {json.dumps(response_data)}\n\n"
                return

            # For content queries, find relevant chunks
            intent = conversation_manager.detect_intent(processed_query)
            relevant_chunks = find_relevant_chunks(processed_query)
            user_message.relevant_document_chunks.set(relevant_chunks)

            # Send relevant chunks in the initial response
            response_data["event"] = "documents"
            response_data["relevant_documents"] = [
                {
                    "title": chunk.document.title,
                    "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk in relevant_chunks
            ]
            yield f"data: {json.dumps(response_data)}\n\n"

            # Find specific articles if mentioned
            specific_articles = self._find_specific_articles(processed_query)
            if specific_articles:
                response_data["event"] = "articles"
                response_data["articles"] = [
                    {
                        "article_num": article['article_num'],
                        "content": article['content'][:200] + "..." if len(article['content']) > 200 else article[
                            'content']
                    }
                    for article in specific_articles
                ]
                yield f"data: {json.dumps(response_data)}\n\n"

            # Generate response
            conversation_history = self._get_conversation_history(conversation)
            is_first_interaction = len(conversation_history) <= 1

            # Choose LLM method based on settings
            if self.use_api_llm:
                # For API LLMs, implement token-by-token streaming if supported
                if self.api_provider == 'anthropic':
                    # Anthropic Claude streaming implementation
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

                        # Create system prompt
                        system_prompt = self._create_system_prompt(relevant_chunks, conversation_history,
                                                                   detected_language)

                        # Setup messages
                        messages = [{"role": "system", "content": system_prompt}]

                        # Add conversation history
                        if conversation_history:
                            for msg in conversation_history[-5:]:
                                messages.append({"role": msg["role"], "content": msg["content"]})

                        # Add current query
                        messages.append({"role": "user", "content": processed_query})

                        # Select model based on complexity
                        model = "claude-3-haiku-20240307"
                        if len(processed_query.split()) > 20 or len(system_prompt) > 1000:
                            model = "claude-3-sonnet-20240229"

                        # Start the streaming response
                        with client.messages.stream(
                                model=model,
                                max_tokens=1024,
                                messages=messages,
                                temperature=0.7
                        ) as stream:
                            # Create an empty assistant message to store the stream
                            assistant_message = Message.objects.create(
                                conversation=conversation,
                                role='assistant',
                                content=""
                            )

                            # Setup streaming
                            full_response = ""
                            response_data["event"] = "message"
                            response_data["message_id"] = assistant_message.id

                            # Process the stream
                            for text in stream.text_stream:
                                full_response += text
                                response_data["content"] = full_response
                                yield f"data: {json.dumps(response_data)}\n\n"

                            # Update the message with the full response
                            assistant_message.content = full_response
                            assistant_message.save()

                            # Add relevant chunks to the message
                            assistant_message.relevant_document_chunks.set(relevant_chunks)

                    except Exception as e:
                        logger.error(f"Error in streaming with Anthropic: {e}")
                        response_text = self._generate_response_with_api_llm(
                            processed_query, relevant_chunks, conversation_history, detected_language
                        )

                        # Send the full response as fallback
                        enhanced_response = conversation_manager.enhance_response(
                            response_text, processed_query, intent, conversation
                        )

                        assistant_message = Message.objects.create(
                            conversation=conversation,
                            role='assistant',
                            content=enhanced_response
                        )

                        response_data["event"] = "message"
                        response_data["message_id"] = assistant_message.id
                        response_data["content"] = enhanced_response
                        yield f"data: {json.dumps(response_data)}\n\n"

                else:
                    # Non-streaming fallback for other API providers
                    response_text = self._generate_response_with_api_llm(
                        processed_query, relevant_chunks, conversation_history, detected_language
                    )

                    enhanced_response = conversation_manager.enhance_response(
                        response_text, processed_query, intent, conversation
                    )

                    assistant_message = Message.objects.create(
                        conversation=conversation,
                        role='assistant',
                        content=enhanced_response
                    )
                    assistant_message.relevant_document_chunks.set(relevant_chunks)

                    response_data["event"] = "message"
                    response_data["message_id"] = assistant_message.id
                    response_data["content"] = enhanced_response
                    yield f"data: {json.dumps(response_data)}\n\n"
            else:
                # Local LLM (non-streaming for now)
                response_text = self._generate_response_with_local_llm(
                    processed_query, relevant_chunks, conversation_history, detected_language
                )

                enhanced_response = conversation_manager.enhance_response(
                    response_text, processed_query, intent, conversation
                )

                assistant_message = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=enhanced_response
                )
                assistant_message.relevant_document_chunks.set(relevant_chunks)

                response_data["event"] = "message"
                response_data["message_id"] = assistant_message.id
                response_data["content"] = enhanced_response
                yield f"data: {json.dumps(response_data)}\n\n"

            # Send completion event
            response_data["event"] = "done"
            yield f"data: {json.dumps(response_data)}\n\n"

            # Log performance
            processing_time = time.time() - start_time
            logger.info(f"Processed streaming query in {processing_time:.2f}s")

        except Exception as e:
            # Log error
            logger.error(f"Error in streaming response: {str(e)}", exc_info=True)

            # Determine appropriate language for error message
            try:
                error_language = self.detect_language(query)
            except:
                error_language = 'fr'  # Default to French

            # Return graceful error response in detected language
            if error_language == 'en':
                error_msg = "I'm sorry, I encountered a technical difficulty. Could you rephrase your question or try again later?"
            elif error_language == 'ar':
                error_msg = "آسف، لقد واجهت صعوبة فنية. هل يمكنك إعادة صياغة سؤالك أو المحاولة مرة أخرى لاحقًا؟"
            else:
                error_msg = "Je suis désolé, j'ai rencontré une difficulté technique. Pourriez-vous reformuler votre question ou réessayer plus tard?"

            # Send error event
            error_data = {
                "event": "error",
                "content": error_msg,
                "language": error_language
            }
            yield f"data: {json.dumps(error_data)}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

    def regenerate_response(self, conversation, message_id):
        """
        Regenerate a response for a specific message with different parameters
        """
        try:
            # Retrieve the message to regenerate
            try:
                message = Message.objects.get(id=message_id, conversation=conversation)

                # Find the user message that triggered this response
                user_messages = conversation.messages.filter(
                    role='user',
                    created_at__lt=message.created_at
                ).order_by('-created_at')

                if not user_messages.exists():
                    return {
                        "error": "Could not find the user message that triggered this response"
                    }

                user_message = user_messages.first()
                query = user_message.content

            except Message.DoesNotExist:
                return {
                    "error": "Message not found in this conversation"
                }

            # Process the query with different parameters
            # Use higher temperature for more creativity
            temp_self = self
            temp_self.response_times = []

            # Get more relevant chunks this time
            processed_query = preprocess_query(query)
            relevant_chunks = find_relevant_chunks(processed_query, top_k=8)  # Get more chunks

            # Get conversation history
            conversation_history = self._get_conversation_history(conversation)

            detected_language = self.detect_language(processed_query)

            # Generate response with more creativity
            if self.use_api_llm:
                if self.api_provider == 'anthropic':
                    # Use a different model or temperature for regeneration
                    import anthropic
                    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

                    # Create enhanced system prompt
                    system_prompt = self._create_system_prompt(relevant_chunks, conversation_history, detected_language)
                    system_prompt += "\n\nIMPORTANT: Provide a fresh perspective on this question, with a different structure and examples than your previous answer."

                    messages = [
                        {"role": "system", "content": system_prompt},
                    ]

                    # Add conversation history
                    if conversation_history:
                        for msg in conversation_history[-5:]:
                            messages.append({"role": msg["role"], "content": msg["content"]})

                    # Add the current query
                    messages.append({"role": "user", "content": processed_query})

                    # Use a more capable model for regeneration
                    model = "claude-3-sonnet-20240229"

                    response = client.messages.create(
                        model=model,
                        max_tokens=1500,  # More tokens for a more detailed response
                        temperature=0.8,  # Higher temperature for more variety
                        messages=messages
                    )

                    response_text = response.content[0].text

                else:  # OpenAI
                    import openai
                    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

                    # Create enhanced system prompt
                    system_prompt = self._create_system_prompt(relevant_chunks, conversation_history, detected_language)
                    system_prompt += "\n\nIMPORTANT: Provide a fresh perspective on this question, with a different structure and examples than your previous answer."

                    messages = [
                        {"role": "system", "content": system_prompt},
                    ]

                    # Add conversation history
                    if conversation_history:
                        for msg in conversation_history[-5:]:
                            messages.append({"role": msg["role"], "content": msg["content"]})

                    # Add the current query
                    messages.append({"role": "user", "content": processed_query})

                    # Use a more capable model for regeneration
                    model = "gpt-4"

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1500,
                        temperature=0.8
                    )

                    response_text = response.choices[0].message.content
            else:
                # Local LLM with different parameters
                response_text = self._generate_response_with_local_llm(
                    processed_query, relevant_chunks, conversation_history, detected_language
                )

            # Now enhance the response with conversation elements
            intent = conversation_manager.detect_intent(processed_query)
            enhanced_response = conversation_manager.enhance_response(
                response_text, processed_query, intent, conversation
            )

            # Update the existing message
            message.content = enhanced_response
            message.save()

            # Update the relevant chunks
            message.relevant_document_chunks.set(relevant_chunks)

            return {
                "message_id": message.id,
                "content": enhanced_response,
                "relevant_documents": [
                    {
                        "title": chunk.document.title,
                        "excerpt": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    }
                    for chunk in relevant_chunks
                ],
                "language": detected_language
            }

        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error regenerating message: {str(e)}", exc_info=True)

            return {
                "error": f"Failed to regenerate response: {str(e)}"
            }

    def upload_and_process_document(self, file_path, title=None):
        """
        Upload and process a new document for the chatbot
        """
        try:
            from .utils import extract_text_from_pdf, store_document_and_chunks

            # Extract text from document
            text = extract_text_from_pdf(file_path)

            if not text:
                return {
                    "error": "Could not extract text from document. Please ensure it's a valid PDF."
                }

            # Generate a title if not provided
            if not title:
                # Try to extract a title from the content
                lines = text.split('\n')
                for i in range(min(10, len(lines))):
                    line = lines[i].strip()
                    if len(line) > 10 and len(line) < 100:
                        title = line
                        break

                # Use filename as fallback
                if not title:
                    import os
                    title = os.path.basename(file_path)

            # Store the document and create chunks
            document = store_document_and_chunks(
                title=title,
                text=text,
                file_path=file_path
            )

            # Analyze document structure
            structure = analyze_document_structure(document.id)

            # Cache the document structure
            if structure and 'articles' in structure:
                self.article_cache[document.id] = structure['articles']

            return {
                "success": True,
                "document_id": document.id,
                "title": document.title,
                "article_count": len(structure['articles']) if structure and 'articles' in structure else 0,
                "chunk_count": document.chunks.count()
            }

        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}", exc_info=True)

            return {
                "error": f"Failed to process document: {str(e)}"
            }

    def analyze_conversation(self, conversation):
        """
        Analyze a conversation to extract insights, topics, and suggestions
        """
        try:
            messages = conversation.messages.all().order_by('created_at')

            if not messages:
                return {
                    "error": "No messages found in this conversation"
                }

            total_messages = messages.count()
            user_messages = messages.filter(role='user').count()
            assistant_messages = messages.filter(role='assistant').count()

            # Extract topics
            topics = set()
            for message in messages.filter(role='user'):
                intent = conversation_manager.detect_intent(message.content)
                if intent not in ['greeting', 'how_are_you', 'thank_you', 'goodbye']:
                    topics.add(intent)

            # Calculate average message length
            total_length = sum(len(message.content) for message in messages)
            avg_length = total_length / total_messages if total_messages > 0 else 0

            # Count article references
            article_references = []
            for message in messages:
                article_mentions = re.findall(r'article\s+(\d+)', message.content.lower())
                article_references.extend(article_mentions)

            # Count unique articles referenced
            unique_articles = set(article_references)

            # Generate analysis summary
            if self.use_api_llm and self.api_provider == 'anthropic':
                # Use Claude to generate a summary
                import anthropic
                client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

                # Prepare conversation transcript
                transcript = ""
                for i, message in enumerate(messages):
                    role = "User" if message.role == 'user' else "Assistant"
                    transcript += f"{role}: {message.content}\n\n"

                # Truncate if too long
                if len(transcript) > 10000:
                    transcript = transcript[:10000] + "...\n\n[Conversation truncated due to length]"

                prompt = f"""Please analyze this conversation transcript between a user and an AI assistant specialized in Tunisian association law.

Conversation transcript:
{transcript}

Provide a brief analysis covering:
1. Main topics discussed
2. Key questions asked by the user
3. Specific legal articles referenced or discussed
4. Areas where the user seemed confused or unsatisfied
5. Suggestions for improving future conversations on this topic

Format your response as a structured report with clear sections."""

                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                analysis_summary = response.content[0].text
            else:
                # Create a simple text summary
                topic_list = ", ".join(topics) if topics else "No specific topics identified"
                article_list = ", ".join(unique_articles) if unique_articles else "No specific articles referenced"

                analysis_summary = f"""Conversation Analysis:

Duration: {(messages.last().created_at - messages.first().created_at).total_seconds() / 60:.1f} minutes
Messages: {total_messages} total ({user_messages} user, {assistant_messages} assistant)
Average message length: {avg_length:.1f} characters

Main topics discussed: {topic_list}
Articles referenced: {article_list}

This conversation covered {len(topics)} distinct topics related to Tunisian association law."""

            return {
                "message_count": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "topics": list(topics),
                "article_references": list(unique_articles),
                "analysis": analysis_summary
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}", exc_info=True)

            return {
                "error": f"Failed to analyze conversation: {str(e)}"
            }