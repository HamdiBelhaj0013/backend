from rest_framework import serializers
from .models import Conversation, Message, Document, DocumentChunk, FeedbackLog, ChatbotSettings


class DocumentChunkSerializer(serializers.ModelSerializer):
    document_title = serializers.CharField(source='document.title', read_only=True)

    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_id', 'content', 'document_title']


class MessageSerializer(serializers.ModelSerializer):
    relevant_chunks = DocumentChunkSerializer(source='relevant_document_chunks', many=True, read_only=True)
    feedback = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'created_at', 'relevant_chunks', 'feedback']

    def get_feedback(self, obj):
        """Get the latest feedback for a message if it exists"""
        latest_feedback = obj.feedback.order_by('-created_at').first()
        if latest_feedback:
            return {
                'rating': latest_feedback.rating,
                'comment': latest_feedback.comment
            }
        return None


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_active', 'messages']


class ConversationListSerializer(serializers.ModelSerializer):
    last_message = serializers.SerializerMethodField()
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_active', 'last_message', 'message_count']

    def get_last_message(self, obj):
        last_message = obj.messages.order_by('created_at').last()
        if last_message:
            return {
                'content': last_message.content[:100] + '...' if len(
                    last_message.content) > 100 else last_message.content,
                'role': last_message.role,
                'created_at': last_message.created_at
            }
        return None

    def get_message_count(self, obj):
        """Get the count of messages in the conversation"""
        return obj.messages.count()


class ChatMessageSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = ['id', 'title', 'language', 'file', 'created_at', 'updated_at', 'chunk_count']

    def get_chunk_count(self, obj):
        """Get the number of chunks for this document"""
        return obj.chunks.count()


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeedbackLog
        fields = ['id', 'message', 'rating', 'comment', 'created_at']
        read_only_fields = ['created_at']


class ChatbotSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatbotSettings
        fields = [
            'id', 'name', 'greeting_message', 'farewell_message',
            'model_path', 'temperature', 'max_tokens', 'memory_window',
            'enable_conversation_enhancements', 'enable_context_aware_responses',
            'enable_fallback_responses', 'updated_at'
        ]
        read_only_fields = ['updated_at']