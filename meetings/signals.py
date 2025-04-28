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
    if created or instance.tracker.has_changed('start_date'):
        from django.db import transaction

        # Schedule reminder based on meeting settings
        reminder_date = instance.start_date - timedelta(days=instance.reminder_days_before)

        # If reminder date is in the future, schedule it
        if reminder_date > timezone.now():
            # We would integrate with a task queue like Celery here
            # For now, just print debug info
            print(f"Scheduled reminder for meeting: {instance.title}")
            print(f"Reminder will be sent on: {reminder_date}")


@receiver(post_save, sender=MeetingNotification)
def send_email_notification(sender, instance, created, **kwargs):
    """
    Send email notification when a platform notification is created
    """
    if created and instance.method in ['email', 'both'] and not instance.is_sent:
        try:
            # Create context for email
            meeting = instance.meeting
            user = instance.user

            # Get attendee info if available in extra_data
            attendee = None
            if instance.extra_data and 'attendee_id' in instance.extra_data:
                try:
                    attendee_id = instance.extra_data['attendee_id']
                    attendee = MeetingAttendee.objects.get(id=attendee_id)
                    print(f"Found attendee with ID {attendee_id}")
                except MeetingAttendee.DoesNotExist:
                    print(f"Attendee with ID {attendee_id} not found")
                    # If attendee not found by ID, try to find by user's email
                    try:
                        member = Member.objects.get(email=user.email, association=meeting.association)
                        attendee = MeetingAttendee.objects.get(meeting=meeting, member=member)
                        print(f"Found alternative attendee with ID {attendee.id}")
                    except (Member.DoesNotExist, MeetingAttendee.DoesNotExist):
                        print(f"No alternative attendee found for {user.email}")
            else:
                # If extra_data doesn't have attendee_id, try to find by user's email
                try:
                    member = Member.objects.get(email=user.email, association=meeting.association)
                    attendee = MeetingAttendee.objects.get(meeting=meeting, member=member)
                    print(f"Found attendee by email lookup: {attendee.id}")
                except (Member.DoesNotExist, MeetingAttendee.DoesNotExist):
                    print(f"No attendee found by email lookup for {user.email}")

            context = {
                'user': user,
                'meeting': meeting,
                'notification': instance,
                'title': instance.title,
                'message': instance.message,
                'meeting_date': meeting.start_date.strftime('%A, %B %d, %Y'),
                'meeting_time': meeting.start_date.strftime('%I:%M %p'),
                'meeting_location': meeting.location if not meeting.is_virtual else 'Virtual Meeting',
                'meeting_link': meeting.meeting_link if meeting.is_virtual else None,
                'attendee': attendee,  # Include the attendee if found
            }

            # Debug output
            print(f"Sending email to {user.email}")
            print(f"Attendee included: {attendee is not None}")
            if attendee:
                print(f"Attendee ID: {attendee.id}, Member: {attendee.member.name}")

            # Render email templates
            html_message = render_to_string("meetings/email_notification.html", context=context)
            plain_message = strip_tags(html_message)

            # Send the email
            msg = EmailMultiAlternatives(
                subject=instance.title,
                body=plain_message,
                from_email="noreply.myorg@gmail.com",
                to=[user.email]
            )

            msg.attach_alternative(html_message, "text/html")
            msg.send()

            # Update notification status
            instance.is_sent = True
            instance.sent_at = timezone.now()
            instance.save()

            print(f"Email notification sent to {user.email}: {instance.title}")

        except Exception as e:
            print(f"Error sending email notification: {str(e)}")
            import traceback
            traceback.print_exc()


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

            # Track if any field has changed
            instance.tracker['has_changed'] = lambda field: field in instance.tracker['changed_fields']

    except Meeting.DoesNotExist:
        # This is a new meeting
        pass
    except Exception as e:
        print(f"Error in meeting tracking: {str(e)}")