from rest_framework import serializers
from .models import Meeting, MeetingAttendee, MeetingReport, MeetingNotification
from api.serializers import MemberSerializer
from users.serializers import UserProfileSerializer


class MeetingAttendeeSerializer(serializers.ModelSerializer):
    member_details = MemberSerializer(source='member', read_only=True)

    class Meta:
        model = MeetingAttendee
        fields = [
            'id', 'meeting', 'member', 'member_details', 'status',
            'attendance_mode', 'notes', 'special_role', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class MeetingReportSerializer(serializers.ModelSerializer):
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    approved_by_details = UserProfileSerializer(source='approved_by', read_only=True)

    class Meta:
        model = MeetingReport
        fields = [
            'id', 'meeting', 'title', 'summary', 'report_file',
            'is_approved', 'approved_by', 'approved_by_details', 'approval_date',
            'created_by', 'created_by_details', 'created_at', 'updated_at'
        ]
        read_only_fields = ['approved_by', 'approval_date', 'created_at', 'updated_at', 'created_by']


class MeetingNotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = MeetingNotification
        fields = [
            'id', 'meeting', 'user', 'title', 'message', 'notification_type',
            'method', 'is_sent', 'is_read', 'sent_at', 'read_at', 'created_at'
        ]
        read_only_fields = ['is_sent', 'sent_at', 'is_read', 'read_at', 'created_at']


class MeetingSerializer(serializers.ModelSerializer):
    created_by_details = UserProfileSerializer(source='created_by', read_only=True)
    association_name = serializers.CharField(source='association.name', read_only=True)
    attendees_count = serializers.SerializerMethodField()
    attendees = MeetingAttendeeSerializer(many=True, read_only=True)
    reports = MeetingReportSerializer(many=True, read_only=True)

    class Meta:
        model = Meeting
        fields = [
            'id', 'title', 'description', 'meeting_type', 'status',
            'association', 'association_name', 'created_by', 'created_by_details',
            'start_date', 'end_date', 'location', 'is_virtual', 'meeting_link',
            'agenda', 'agenda_document',
            'is_recurring', 'recurrence_pattern', 'notification_method',
            'reminder_days_before', 'created_at', 'updated_at',
            'attendees_count', 'attendees', 'reports'
        ]
        read_only_fields = ['created_at', 'updated_at', 'created_by', 'attendees_count']

    def get_attendees_count(self, obj):
        return obj.attendees.filter(status='present').count()

    def validate(self, data):
        # Validate start and end dates
        if 'start_date' in data and 'end_date' in data:
            if data['end_date'] < data['start_date']:
                raise serializers.ValidationError("End date cannot be before start date")

        # Validate virtual meeting requirements
        if data.get('is_virtual', False) and not data.get('meeting_link'):
            raise serializers.ValidationError("Virtual meetings must have a meeting link")

        return data

    def create(self, validated_data):
        # Set the creator to the current user
        validated_data['created_by'] = self.context['request'].user

        # Handle recurring meeting pattern if needed
        if validated_data.get('is_recurring') and not validated_data.get('recurrence_pattern'):
            # Set default monthly recurrence if not specified
            validated_data['recurrence_pattern'] = {
                'frequency': 'monthly',
                'interval': 1,
                'day_of_month': validated_data['start_date'].day,
                'end_after': 12  # default to recurring for 12 months
            }

        return super().create(validated_data)


class MeetingListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing meetings"""
    association_name = serializers.CharField(source='association.name', read_only=True)
    attendees_count = serializers.SerializerMethodField()
    is_upcoming = serializers.BooleanField(read_only=True)
    has_report = serializers.SerializerMethodField()

    class Meta:
        model = Meeting
        fields = [
            'id', 'title', 'meeting_type', 'status', 'association_name',
            'start_date', 'end_date', 'location', 'is_virtual',
            'created_at', 'attendees_count', 'is_upcoming', 'has_report'
        ]

    def get_attendees_count(self, obj):
        return obj.attendees.filter(status='present').count()

    def get_has_report(self, obj):
        return obj.reports.exists()


class ReportGenerationSerializer(serializers.Serializer):
    """Serializer for generating meeting reports"""
    meeting_id = serializers.IntegerField()
    include_attendance = serializers.BooleanField(default=True)
    summary = serializers.CharField(required=False, allow_blank=True)
    report_title = serializers.CharField(required=False, allow_blank=True)