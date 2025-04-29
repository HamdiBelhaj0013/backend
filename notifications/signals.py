from django.db.models.signals import post_save, post_delete, pre_save
from django.dispatch import receiver
from django.utils import timezone
from django.apps import apps
from django.db import transaction
from django.contrib.auth import get_user_model

from .services import NotificationService

User = get_user_model()  # This gets your CustomUser model


@receiver(post_save, sender='finances.Transaction')
def handle_transaction_save(sender, instance, created, **kwargs):
    """
    Signal to handle transaction creation and updates

    Creates appropriate notifications when transactions are created or updated
    """
    # Check if this is a new transaction
    if created:
        # Process the transaction to see if it requires official letter
        NotificationService.process_critical_transaction(
            transaction=instance,
            action_type="create"
        )
    else:
        # Update action
        NotificationService.process_critical_transaction(
            transaction=instance,
            action_type="update"
        )


@receiver(post_delete, sender='finances.Transaction')
def handle_transaction_delete(sender, instance, **kwargs):
    """
    Signal to handle transaction deletion

    Creates appropriate notifications when transactions are deleted
    """
    NotificationService.process_critical_transaction(
        transaction=instance,
        action_type="delete"
    )


@receiver(post_save, sender='meetings.Meeting')
def handle_meeting_save(sender, instance, created, **kwargs):
    """
    Signal to handle meeting creation and updates

    Creates appropriate notifications when meetings are created, updated or cancelled
    """
    Meeting = apps.get_model('meetings', 'Meeting')

    # For new meetings
    if created:
        NotificationService.send_meeting_notification(
            meeting=instance,
            notification_type="meeting_scheduled"
        )
        return

    # Try to detect if meeting was cancelled
    # Check if instance is being or has been saved
    try:
        old_instance = Meeting.objects.get(pk=instance.pk)
        if old_instance.status != 'cancelled' and instance.status == 'cancelled':
            NotificationService.send_meeting_notification(
                meeting=instance,
                notification_type="meeting_cancelled"
            )
    except Meeting.DoesNotExist:
        # This should not happen in post_save signal
        pass


@receiver(post_save, sender='meetings.Meeting')
def schedule_meeting_reminders(sender, instance, created, **kwargs):
    """
    Schedule reminders for upcoming meetings
    """
    # Only process active meetings that are in the future
    if instance.status != 'scheduled' or instance.start_date < timezone.now():
        return

    # Schedule the reminders via a task scheduler (e.g., Celery)
    # For this example, we'll simulate scheduling by just logging it

    # Reminder 1 day before
    reminder_time = instance.start_date - timezone.timedelta(days=1)

    # Here you would typically create a Celery task scheduled for reminder_time
    # For example:
    # send_meeting_reminder.apply_async(args=[instance.id], eta=reminder_time)

    # But for now, we'll just log it
    print(f"Scheduled reminder for meeting {instance.id} at {reminder_time}")


@receiver(post_save, sender='meetings.Meeting')
def check_meeting_report(sender, instance, created, **kwargs):
    """
    Check if meeting needs a report after it ends
    """
    # Only process meetings that have ended and don't already have reports
    if instance.status != 'completed' or instance.end_date > timezone.now():
        return

    # Check if meeting already has a report (would need to adapt this to your model structure)
    Report = apps.get_model('meetings', 'Report', require_ready=False)

    try:
        has_report = Report.objects.filter(meeting=instance).exists()
    except:
        # Model might not exist yet
        has_report = False

    if not has_report:
        # Schedule a notification for 3 days after the meeting
        report_due_date = instance.end_date + timezone.timedelta(days=3)

        # If 3 days have already passed
        if timezone.now() > report_due_date:
            NotificationService.send_report_due_notification(instance)
        else:
            # Here you would schedule the notification for later
            # For example with Celery:
            # send_report_notification.apply_async(args=[instance.id], eta=report_due_date)
            pass


@receiver(post_save, sender=User)
def handle_user_membership(sender, instance, created, **kwargs):
    """
    Signal to handle user joining or role changes
    """
    if created:
        NotificationService.send_to_role(
            role='admin',
            title=f"New user joined: {instance.get_full_name() or instance.email}",
            message=f"A new user {instance.get_full_name() or instance.email} has joined the association.",
            notification_type="user_joined",
            related_object=instance,
            url="/members",
            priority='medium'
        )
    else:
        # Check if role has changed - would need to adapt this check to your specific user model
        try:
            old_instance = sender.objects.get(pk=instance.pk)
            if hasattr(old_instance, 'role') and hasattr(instance, 'role'):
                if old_instance.role != instance.role:
                    NotificationService.send_to_role(
                        role='admin',
                        title=f"User role changed: {instance.get_full_name() or instance.email}",
                        message=(
                            f"User {instance.get_full_name() or instance.email} role changed "
                            f"from {old_instance.role} to {instance.role}."
                        ),
                        notification_type="user_joined",
                        related_object=instance,
                        url="/members",
                        priority='medium'
                    )
        except sender.DoesNotExist:
            pass


@receiver(post_delete, sender=User)
def handle_user_deletion(sender, instance, **kwargs):
    """
    Signal to handle user deletion
    """
    NotificationService.send_to_role(
        role='admin',
        title=f"User left: {instance.get_full_name() or instance.email}",
        message=f"User {instance.get_full_name() or instance.email} has been removed from the association.",
        notification_type="user_left",
        url="/members",
        priority='medium'
    )