from rest_framework import viewsets, status, views
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.settings import api_settings
from django.shortcuts import get_object_or_404
from .models import Conversation, Message, Document, FeedbackLog, ChatbotSettings
from .serializers import ConversationSerializer, ConversationListSerializer, ChatMessageSerializer, DocumentSerializer
from .services import ChatbotService
from .utils import find_relevant_chunks
from .conversation_handlers import conversation_manager

# Initialize the chatbot service
chatbot_service = ChatbotService()


class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for managing conversations with the chatbot"""
    serializer_class = ConversationSerializer
    permission_classes = [AllowAny]  # Adjust based on your authentication requirements

    # Add this property to fix the settings access issue
    @property
    def settings(self):
        return api_settings

    def get_queryset(self):
        """Filter conversations for the current user"""
        if self.request.user.is_authenticated:
            return Conversation.objects.filter(user=self.request.user).order_by('-updated_at')
        return Conversation.objects.none()

    def get_serializer_class(self):
        """Use different serializers for list vs detail views"""
        if self.action == 'list':
            return ConversationListSerializer
        return self.serializer_class

    def create(self, request, *args, **kwargs):
        """Create a new conversation"""
        conversation = Conversation.objects.create(
            title="New Conversation",
            user=self.request.user if self.request.user.is_authenticated else None
        )
        return Response(self.get_serializer(conversation).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def chat(self, request, pk=None):
        """Send a message to the chatbot and get a response"""
        conversation = self.get_object()

        # Validate input
        serializer = ChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Get user message
        user_message = serializer.validated_data['message']

        # Process message using the chatbot service
        response = chatbot_service.process_message(conversation, user_message)

        # Update conversation timestamp
        conversation.save()  # This will update the updated_at field

        return Response(response)

    @action(detail=True, methods=['post'])
    def feedback(self, request, pk=None):
        """Submit feedback for a message"""
        message_id = request.data.get('message_id')
        rating = request.data.get('rating')
        comment = request.data.get('comment', '')

        if not message_id or not rating or rating not in range(1, 6):
            return Response(
                {"error": "Valid message_id and rating (1-5) are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            message = Message.objects.get(id=message_id, conversation_id=pk)

            # Record feedback
            FeedbackLog.objects.create(
                message=message,
                rating=rating,
                comment=comment
            )

            return Response({"status": "Feedback recorded"})

        except Message.DoesNotExist:
            return Response(
                {"error": "Message not found in this conversation"},
                status=status.HTTP_404_NOT_FOUND
            )

    @action(detail=False, methods=['get'])
    def get_chatbot_settings(self, request):
        """Get chatbot settings (greeting, name, etc.)"""
        settings = ChatbotSettings.get_settings()
        return Response({
            "name": settings.name,
            "greeting": settings.greeting_message,
            "farewell": settings.farewell_message,
        })


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for managing reference documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]

    # Add this property to fix the settings access issue
    @property
    def settings(self):
        return api_settings

    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        """Process a document to extract and chunk its content"""
        document = self.get_object()

        if not document.file:
            return Response(
                {"error": "No file attached to this document"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            from .utils import process_policy_document
            result = process_policy_document(document.file.path)
            return Response({
                "status": "success",
                "chunks_created": result.chunks.count() if hasattr(result, 'chunks') else 0
            })
        except Exception as e:
            return Response(
                {"error": f"Error processing document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DirectChatView(views.APIView):
    """View for direct testing without conversations"""
    permission_classes = [AllowAny]

    # Add this property to fix the settings access issue
    @property
    def settings(self):
        return api_settings

    def post(self, request):
        # Extract query from request
        query = request.data.get('query', '')
        if not query:
            return Response({"error": "No query provided. Please include a 'query' field."}, status=400)

        # Check if it's a purely conversational query first
        conversation_response, is_conversational = conversation_manager.handle_conversation(query)

        if is_conversational:
            # For conversational queries, we don't need to search for document chunks
            return Response({
                "query": query,
                "response": conversation_response,
                "relevant_chunks": [],
                "is_conversational": True
            })

        # For non-conversational queries, find relevant chunks
        relevant_chunks = find_relevant_chunks(query)

        # Use rule-based responses with relevant chunks
        response_text = self._generate_rule_based_response(query, relevant_chunks)

        # Return response
        return Response({
            "query": query,
            "response": response_text,
            "relevant_chunks": [
                {"content": chunk.content[:100] + "..."}
                for chunk in relevant_chunks[:3]
            ],
            "is_conversational": False
        })

    def _generate_rule_based_response(self, query, relevant_chunks):
        """Generate a rule-based response with relevant context"""
        query_lower = query.lower()

        # Create responses for common questions with context awareness
        if any(word in query_lower for word in ["créer", "constituer", "fonder"]) and "association" in query_lower:
            return """Pour créer une association en Tunisie selon le décret-loi n° 2011-88, vous devez:

1. Adresser une lettre recommandée au secrétaire général du gouvernement contenant:
   - Une déclaration avec le nom, l'objet, les objectifs et le siège de l'association
   - Copies des cartes d'identité des fondateurs
   - Deux exemplaires des statuts signés

2. Après réception de l'accusé, déposer une annonce à l'Imprimerie Officielle dans les 7 jours.

L'association est légalement constituée dès l'envoi de la lettre et acquiert la personnalité morale après publication au Journal Officiel."""

        elif any(word in query_lower for word in ["statut", "statuts"]):
            return """Selon l'article 10 du décret-loi n° 2011-88, les statuts d'une association doivent contenir:

1. La dénomination officielle en arabe et éventuellement en langue étrangère
2. L'adresse du siège principal
3. Une présentation des objectifs et des moyens de réalisation
4. Les conditions d'adhésion et les droits/obligations des membres
5. L'organigramme, le mode d'élection et les prérogatives des organes
6. L'organe responsable des modifications et des décisions de dissolution/fusion
7. Les modes de prise de décision et de règlement des différends
8. Le montant de la cotisation s'il existe"""

        elif any(word in query_lower for word in ["finance", "financement", "ressource", "budget", "argent"]):
            return """Selon l'article 34 du décret-loi n° 2011-88, les ressources d'une association se composent de:

1. Les cotisations de ses membres
2. Les aides publiques
3. Les dons, donations et legs d'origine nationale ou étrangère
4. Les recettes résultant de ses biens, activités et projets

L'association est tenue de consacrer ses ressources aux activités nécessaires à la réalisation de ses objectifs (article 37).

Il est interdit d'accepter des aides d'États n'ayant pas de relations diplomatiques avec la Tunisie (article 35)."""

        elif "dissolution" in query_lower:
            return """La dissolution d'une association selon le décret-loi n° 2011-88 peut être:

1. Volontaire: par décision de ses membres conformément aux statuts
2. Judiciaire: par jugement du tribunal

En cas de dissolution volontaire, l'association doit:
- Informer le secrétaire général dans les 30 jours suivant la décision
- Désigner un liquidateur judiciaire
- Présenter un état de ses biens pour s'acquitter de ses obligations

Le reliquat sera distribué selon les statuts ou attribué à une association similaire."""

        elif any(word in query_lower for word in ["membre", "adhérer", "adhésion"]):
            return """Selon l'article 17 du décret-loi n° 2011-88, un membre d'association doit:

1. Être de nationalité tunisienne ou résident en Tunisie
2. Avoir au moins 13 ans
3. Accepter par écrit les statuts de l'association
4. Verser la cotisation requise

Les fondateurs et dirigeants ne peuvent pas être en charge de responsabilités dans des partis politiques (article 9).

Les membres et salariés doivent éviter les conflits d'intérêts (article 18)."""

        else:
            # Try to extract information from the relevant chunks
            if relevant_chunks:
                # Combine information from chunks for a more informed response
                combined_info = "\n\n".join([chunk.content for chunk in relevant_chunks[:2]])

                return f"""Selon le décret-loi n° 2011-88 sur les associations en Tunisie, voici ce que je peux vous dire sur votre question:

{combined_info}

Est-ce que cela répond à votre question? Sinon, pourriez-vous la reformuler pour que je puisse mieux vous aider?"""
            else:
                # For other questions without relevant chunks, return a generic response
                return """D'après le décret-loi n° 2011-88 sur les associations en Tunisie, je n'ai pas de réponse spécifique préparée pour cette question.

Les principales catégories d'informations disponibles concernent:
- La création d'associations
- Les statuts d'associations
- Le financement et les ressources
- La dissolution d'associations
- Les conditions d'adhésion et les membres

Pourriez-vous reformuler votre question dans l'une de ces catégories? Je serai ravi de vous aider."""

    def unload_model(self):
        """Unload the model to free memory"""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            import gc
            gc.collect()
            self.llm = None
            self.model_loaded = False