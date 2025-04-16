from rest_framework import serializers
from .models import Conversation, Message, Document, DocumentChunk


class DocumentChunkSerializer(serializers.ModelSerializer):
    document_title = serializers.CharField(source='document.title', read_only=True)

    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_id', 'content', 'document_title']


class MessageSerializer(serializers.ModelSerializer):
    relevant_chunks = DocumentChunkSerializer(source='relevant_document_chunks', many=True, read_only=True)

    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'created_at', 'relevant_chunks']


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_active', 'messages']


class ConversationListSerializer(serializers.ModelSerializer):
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_active', 'last_message']

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


class ChatMessageSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'language', 'file', 'created_at', 'updated_at']