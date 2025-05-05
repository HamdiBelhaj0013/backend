from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from .models import Conversation, Message, Document, FeedbackLog, ChatbotSettings
from .serializers import ConversationSerializer, ConversationListSerializer, ChatMessageSerializer, DocumentSerializer
from .services import ChatbotService

# Initialize the chatbot service
chatbot_service = ChatbotService()


class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for managing conversations with the chatbot"""
    serializer_class = ConversationSerializer
    permission_classes = [AllowAny]  # Adjust based on your authentication requirements

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

    def create(self, serializer):
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
    def settings(self, request):
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
    permission_classes = [IsAuthenticated]  # Limit to authenticated users

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