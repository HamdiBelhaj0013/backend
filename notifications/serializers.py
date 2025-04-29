from rest_framework import serializers
from .models import Notification, OfficialLetterLog
from django.contrib.auth import get_user_model

User = get_user_model()


class UserMiniSerializer(serializers.ModelSerializer):
    """Simple user serializer for nested relationships"""

    class Meta:
        model = User
        fields = ['id', 'username', 'full_name', 'role']


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for notification objects"""
    recipient_name = serializers.SerializerMethodField()
    days_since_created = serializers.SerializerMethodField()
    days_until_deadline = serializers.SerializerMethodField()

    class Meta:
        model = Notification
        fields = [
            'id', 'recipient', 'recipient_name', 'recipient_role',
            'title', 'message', 'notification_type',
            'related_object_id', 'related_object_type',
            'url', 'created_at', 'read', 'read_at',
            'priority', 'requires_action', 'action_deadline',
            'action_completed', 'action_completed_at',
            'requires_official_letter', 'official_letter_recipient',
            'official_letter_sent', 'official_letter_sent_at',
            'days_since_created', 'days_until_deadline'
        ]
        read_only_fields = ['created_at', 'read_at', 'action_completed_at', 'official_letter_sent_at']

    def get_recipient_name(self, obj):
        """Get recipient name if available"""
        if obj.recipient:
            return obj.recipient.get_full_name() or obj.recipient.username
        return None

    def get_days_since_created(self, obj):
        """Calculate days since notification was created"""
        from django.utils import timezone
        import datetime

        if obj.created_at:
            delta = timezone.now().date() - obj.created_at.date()
            return delta.days
        return None

    def get_days_until_deadline(self, obj):
        """Calculate days until action deadline"""
        from django.utils import timezone
        import datetime

        if obj.requires_action and obj.action_deadline and not obj.action_completed:
            delta = obj.action_deadline.date() - timezone.now().date()
            return delta.days
        return None


class OfficialLetterLogSerializer(serializers.ModelSerializer):
    """Serializer for official letter logs"""
    sender_details = UserMiniSerializer(source='sender', read_only=True)
    notification_details = NotificationSerializer(source='notification', read_only=True)
    letter_file_url = serializers.SerializerMethodField()
    proof_file_url = serializers.SerializerMethodField()

    class Meta:
        model = OfficialLetterLog
        fields = [
            'id', 'notification', 'notification_details',
            'sender', 'sender_details', 'recipient',
            'subject', 'content', 'reference_number',
            'tracking_number', 'letter_file', 'letter_file_url',
            'date_prepared', 'date_sent', 'proof_of_sending',
            'proof_file_url'
        ]
        read_only_fields = ['date_prepared']

    def get_letter_file_url(self, obj):
        """Get full URL for letter file if available"""
        if obj.letter_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.letter_file.url)
        return None

    def get_proof_file_url(self, obj):
        """Get full URL for proof file if available"""
        if obj.proof_of_sending:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.proof_of_sending.url)
        return None


class NotificationCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating notifications"""
    recipient_id = serializers.IntegerField(required=False, allow_null=True)

    class Meta:
        model = Notification
        fields = [
            'recipient_id', 'recipient_role', 'title', 'message',
            'notification_type', 'related_object_id', 'related_object_type',
            'url', 'priority', 'requires_action', 'action_deadline',
            'requires_official_letter', 'official_letter_recipient'
        ]

    def validate_recipient_id(self, value):
        """Validate that recipient exists"""
        if value:
            try:
                User.objects.get(pk=value)
            except User.DoesNotExist:
                raise serializers.ValidationError("User with this ID does not exist")
        return value

    def create(self, validated_data):
        """Create notification with recipient from ID"""
        recipient_id = validated_data.pop('recipient_id', None)

        if recipient_id:
            recipient = User.objects.get(pk=recipient_id)
            validated_data['recipient'] = recipient

        return Notification.objects.create(**validated_data)


class MarkNotificationReadSerializer(serializers.Serializer):
    """Serializer for marking notifications as read"""
    notification_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False
    )

    def validate(self, data):
        """Validate notification IDs"""
        notification_ids = data.get('notification_ids', [])
        user = self.context['request'].user

        if notification_ids:
            # Verify all notifications exist and belong to the user
            notifications = Notification.objects.filter(pk__in=notification_ids)

            # Check if the user has access to these notifications
            for notification in notifications:
                if notification.recipient and notification.recipient != user:
                    # If notification has a specific recipient, it must be the user
                    raise serializers.ValidationError(
                        f"Notification {notification.id} does not belong to this user"
                    )

        return data