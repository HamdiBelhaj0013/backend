from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone
from .models import Meeting, MeetingNotification, MeetingAttendee
from api.models import Member
from datetime import timedelta
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags


@receiver(post_save, sender=Meeting)
def schedule_meeting_reminders(sender, instance, created, **kwargs):
    """
    Schedule automated reminders for a meeting
    """
    # Check if this is a new meeting or if start_date has changed
    # Fixed to use dictionary access instead of attribute access
    if created or (hasattr(instance, 'tracker') and
                  'changed_fields' in instance.tracker and
                  'start_date' in instance.tracker['changed_fields']):
        from django.db import transaction

        # Schedule reminder based on meeting settings
        reminder_date = instance.start_date - timedelta(days=instance.reminder_days_before)

        # If reminder date is in the future, schedule it
        if reminder_date > timezone.now():
            # We would integrate with a task queue like Celery here
            # For now, just print debug info
            print(f"Scheduled reminder for meeting: {instance.title}")
            print(f"Reminder will be sent on: {reminder_date}")
@receiver(post_save, sender=Meeting)
def meeting_status_changed(sender, instance, **kwargs):
    """Detect when a meeting is marked as completed and send a reminder about the report"""

    # Check if saved instance exists in the database (to avoid post_init signals)
    if not instance.pk:
        return

    # Get previous instance if it exists
    try:
        # Fixed to handle tracker as a dictionary
        if hasattr(instance, 'tracker') and 'old_status' in instance.tracker:
            previous_status = instance.tracker['old_status']
        else:
            previous_status = None
    except Exception:
        previous_status = None

    # If status changed to 'completed' and no report exists yet
    if instance.status == 'completed' and previous_status != 'completed' and not instance.reports.exists():
        # Create a notification for the meeting creator or association admins
        from notifications.services import NotificationService

        NotificationService.send_notification(
            title=f"Procès-Verbal requis: {instance.title}",
            message=(
                f"La réunion '{instance.title}' est maintenant terminée. "
                f"Veuillez joindre un Procès-Verbal (PV) pour finaliser cette réunion."
            ),
            notification_type='report_due',
            related_object=instance,
            url=f"/meetings/{instance.id}/report",
            priority='medium',
            requires_action=True,
            action_deadline=timezone.now() + timezone.timedelta(days=7)  # Due in one week
        )

@receiver(pre_save, sender=Meeting)
def setup_meeting_tracking(sender, instance, **kwargs):
    """
    Set up tracking for meeting changes
    """
    try:
        # Only check if this is an existing meeting
        if instance.pk:
            old_instance = Meeting.objects.get(pk=instance.pk)

            # Store previous values for tracking
            if not hasattr(instance, 'tracker'):
                instance.tracker = {}

            # Track specific fields
            tracked_fields = ['start_date', 'end_date', 'location', 'status']
            instance.tracker['changed_fields'] = []

            for field in tracked_fields:
                old_value = getattr(old_instance, field)
                new_value = getattr(instance, field)

                if old_value != new_value:
                    instance.tracker['changed_fields'].append(field)
                    instance.tracker[f'old_{field}'] = old_value

            # REMOVED: Don't define the lambda function anymore since we're not using it
            # Instead, we check the changed_fields list directly in the other signal handlers

    except Meeting.DoesNotExist:
        # This is a new meeting
        pass
    except Exception as e:
        print(f"Error in meeting tracking: {str(e)}")