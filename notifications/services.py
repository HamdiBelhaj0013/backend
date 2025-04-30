from django.db import transaction
from django.utils import timezone
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse

from .models import Notification, OfficialLetterLog

try:
    # Try to import the role mapper
    from .role_mapper import get_frontend_role, get_backend_role
except ImportError:
    # Fallback if role_mapper.py doesn't exist yet
    def get_frontend_role(role):
        """Map backend roles to frontend roles"""
        role_mapping = {
            'admin': 'president',
            'president': 'president',
            'treasurer': 'treasurer',
            'secretary': 'secretary',
            'member': 'member',
        }
        return role_mapping.get(role, role)


    def get_backend_role(role):
        """Map frontend roles to backend roles"""
        role_mapping = {
            'president': 'admin',
        }
        return role_mapping.get(role, role)

User = get_user_model()


class NotificationService:
    """
    Notification management service
    """

    @staticmethod
    def send_notification(
            title,
            message,
            notification_type,
            recipient=None,
            recipient_role=None,
            related_object=None,
            url=None,
            priority='medium',
            requires_action=False,
            action_deadline=None,
            requires_official_letter=False,
            official_letter_recipient=None
    ):
        """
        Send a notification to a user or role

        Args:
            title (str): Notification title
            message (str): Notification message
            notification_type (str): Notification type
            recipient (User, optional): Specific recipient user
            recipient_role (str, optional): Role of recipients
            related_object (Model, optional): Related model object
            url (str, optional): Redirect URL
            priority (str, optional): Notification priority (low, medium, high)
            requires_action (bool, optional): If action is required
            action_deadline (datetime, optional): Deadline for required action
            requires_official_letter (bool, optional): If official letter is required
            official_letter_recipient (str, optional): Official letter recipient

        Returns:
            Notification: Created notification object
        """
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer
        from django.core.serializers.json import DjangoJSONEncoder
        import json

        related_object_id = None
        related_object_type = None

        if related_object:
            related_object_id = related_object.id
            related_object_type = related_object._meta.model_name

        # Map recipient_role to frontend role name if provided, but don't send duplicate notifications
        if recipient_role:
            mapped_role = get_frontend_role(recipient_role)

            # To prevent duplication, override the original role with the mapped role
            # This ensures consistent role naming throughout the system
            recipient_role = mapped_role

        # Create the notification
        notification = Notification.objects.create(
            recipient=recipient,
            recipient_role=recipient_role,
            title=title,
            message=message,
            notification_type=notification_type,
            related_object_id=related_object_id,
            related_object_type=related_object_type,
            url=url,
            priority=priority,
            requires_action=requires_action,
            action_deadline=action_deadline,
            requires_official_letter=requires_official_letter,
            official_letter_recipient=official_letter_recipient
        )

        # If official letter is required, create a letter log
        if requires_official_letter and official_letter_recipient:
            OfficialLetterLog.objects.create(
                notification=notification,
                recipient=official_letter_recipient,
                subject=title,
                content=message
            )

        # Send real-time notification via WebSocket
        channel_layer = get_channel_layer()

        # Convert notification to dict for JSON serialization
        from .serializers import NotificationSerializer
        notification_data = NotificationSerializer(notification).data

        # Determine which groups should receive this notification
        groups = []

        # If notification has specific recipient, send to their personal group
        if recipient:
            groups.append(f"user_notifications_{recipient.id}")

        # If notification has a role, send to that role group - BUT DON'T DUPLICATE for admin/president
        if recipient_role:
            groups.append(f"role_notifications_{recipient_role}")

            # Do NOT add duplicate groups - this was causing the duplication issue
            # We're using the already-mapped role, so no need to add additional groups

        # If notification is for everyone (no specific recipient or role)
        if not recipient and not recipient_role:
            groups.append("all_notifications")

        # Send to all determined groups
        for group_name in groups:
            try:
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        "type": "notification.message",
                        "notification": notification_data
                    }
                )
            except Exception as e:
                # Log errors but don't interrupt notification creation
                print(f"WebSocket notification send error to {group_name}: {str(e)}")

        return notification

    @staticmethod
    def send_to_all(
            title,
            message,
            notification_type,
            related_object=None,
            url=None,
            priority='medium',
            requires_action=False,
            action_deadline=None,
            requires_official_letter=False,
            official_letter_recipient=None
    ):
        """Send a notification to all users"""
        return NotificationService.send_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=None,  # No specific recipient means all users
            recipient_role=None,  # No specific role means all users
            related_object=related_object,
            url=url,
            priority=priority,
            requires_action=requires_action,
            action_deadline=action_deadline,
            requires_official_letter=requires_official_letter,
            official_letter_recipient=official_letter_recipient
        )

    @staticmethod
    def send_to_role(
            role,
            title,
            message,
            notification_type,
            related_object=None,
            url=None,
            priority='medium',
            requires_action=False,
            action_deadline=None,
            requires_official_letter=False,
            official_letter_recipient=None
    ):
        """Send a notification to all users with the specified role"""
        # Map the role from backend naming to frontend naming for consistency
        # This ensures that notification.recipient_role matches frontend authorization roles
        mapped_role = get_frontend_role(role)

        # For debugging
        if role != mapped_role:
            print(f"Notification role mapped from '{role}' to '{mapped_role}'")

        return NotificationService.send_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=None,  # No specific recipient
            recipient_role=mapped_role,  # Use the mapped role
            related_object=related_object,
            url=url,
            priority=priority,
            requires_action=requires_action,
            action_deadline=action_deadline,
            requires_official_letter=requires_official_letter,
            official_letter_recipient=official_letter_recipient
        )

    @staticmethod
    def send_to_admins(
            title,
            message,
            notification_type,
            related_object=None,
            url=None,
            priority='medium',
            requires_action=False,
            action_deadline=None,
            requires_official_letter=False,
            official_letter_recipient=None
    ):
        """Send a notification to all admin users"""
        return NotificationService.send_to_role(
            role='president',  # Use 'president' instead of 'admin' for consistency
            title=title,
            message=message,
            notification_type=notification_type,
            related_object=related_object,
            url=url,
            priority=priority,
            requires_action=requires_action,
            action_deadline=action_deadline,
            requires_official_letter=requires_official_letter,
            official_letter_recipient=official_letter_recipient
        )

    @staticmethod
    def send_meeting_notification(meeting, notification_type, message=None):
        """
        Send a notification about a meeting

        Args:
            meeting: Meeting object
            notification_type: Meeting notification type
            message: Optional custom message
        """
        title_map = {
            'meeting_scheduled': f"Nouvelle réunion : {meeting.title}",
            'meeting_cancelled': f"Réunion annulée : {meeting.title}",
            'meeting_reminder': f"Rappel : {meeting.title}",
        }

        message_map = {
            'meeting_scheduled': f"Une nouvelle réunion '{meeting.title}' a été programmée pour le {meeting.start_date.strftime('%d/%m/%Y à %H:%M')}",
            'meeting_cancelled': f"La réunion '{meeting.title}' prévue pour le {meeting.start_date.strftime('%d/%m/%Y à %H:%M')} a été annulée",
            'meeting_reminder': f"Rappel : La réunion '{meeting.title}' commencera le {meeting.start_date.strftime('%d/%m/%Y à %H:%M')}",
        }

        # All members should receive meeting notifications
        return NotificationService.send_notification(
            title=title_map.get(notification_type, f"Mise à jour de réunion : {meeting.title}"),
            message=message or message_map.get(notification_type,
                                               f"Mise à jour concernant la réunion : {meeting.title}"),
            notification_type=notification_type,
            related_object=meeting,
            url=f"/meetings/{meeting.id}",
            priority='medium',
        )

    @staticmethod
    def send_report_due_notification(meeting):
        """Send a notification that a meeting report is due"""
        return NotificationService.send_to_role(
            role='president',  # Use 'president' instead of 'admin' for consistency
            title=f"Rapport à remettre pour la réunion : {meeting.title}",
            message=f"Un rapport doit être soumis pour la réunion '{meeting.title}' tenue le {meeting.start_date.strftime('%d/%m/%Y')}",
            notification_type='report_due',
            related_object=meeting,
            url=f"/meetings/{meeting.id}/report",
            priority='high',
            requires_action=True,
            action_deadline=timezone.now() + timezone.timedelta(days=7),  # Due in 7 days
        )

    @staticmethod
    # Fixed send_transaction_notification method

    @staticmethod
    def send_transaction_notification(transaction, notification_type, requires_letter=False):
        """
        Send a notification about a transaction - FIXED VERSION
        This version is role-aware and prevents duplicate notifications

        Args:
            transaction: Transaction object
            notification_type: Transaction notification type
            requires_letter: If it requires an official letter
        """
        # Determine transaction type and amount for the message
        transaction_type = "revenu" if transaction.transaction_type == "income" else "dépense"
        amount = transaction.amount

        title_map = {
            'transaction_created': f"Nouveau {transaction_type} : {amount} TND",
            'transaction_updated': f"{transaction_type.capitalize()} mis à jour : {amount} TND",
            'transaction_deleted': f"{transaction_type.capitalize()} supprimé : {amount} TND",
            'donation_received': f"Don reçu : {amount} TND",
        }

        message_map = {
            'transaction_created': f"Un nouveau {transaction_type} de {amount} TND a été enregistré",
            'transaction_updated': f"Le {transaction_type} de {amount} TND a été mis à jour",
            'transaction_deleted': f"Le {transaction_type} de {amount} TND a été supprimé",
            'donation_received': f"Un don de {amount} TND a été reçu",
        }

        title = title_map.get(notification_type, f"Mise à jour de {transaction_type} : {amount} TND")
        message = message_map.get(notification_type, f"Mise à jour concernant {transaction_type} : {amount} TND")

        # Create a single notification with multiple recipient roles instead of separate notifications
        # This ensures each user gets exactly one notification regardless of their roles
        notification = NotificationService.send_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=None,
            recipient_role="finance",  # Use a generic role for financial notifications
            related_object=transaction,
            url="/finance",
            priority='high' if requires_letter else 'medium',
            requires_official_letter=requires_letter,
            official_letter_recipient="Secrétaire du Premier Ministre" if requires_letter else None
        )

        # If an official letter is required, send a specific notification to the president
        if requires_letter:
            NotificationService.send_to_role(
                role='president',
                title="Lettre Officielle Requise",
                message=f"Une lettre officielle est requise pour la transaction de {amount} TND. Veuillez préparer cette lettre dès que possible.",
                notification_type='official_letter_required',
                related_object=transaction,
                url="/finance/letters",
                priority='high',
                requires_action=True,
                action_deadline=timezone.now() + timezone.timedelta(days=14),
                requires_official_letter=True,
                official_letter_recipient="Premier Ministère"
            )

        return notification

    @staticmethod
    def process_critical_transaction(transaction, action_type):
        """
        Process a critical transaction that requires an official letter

        Args:
            transaction: Transaction object
            action_type: create, update, or delete
        """
        requires_letter = False

        # Check if this transaction requires an official letter
        if transaction.transaction_type == "income" and transaction.category == "donation":
            # External donations require official letters
            if hasattr(transaction, 'donor_details') and getattr(transaction.donor_details, 'is_external', False):
                requires_letter = True

        # For creation and deletion of important transactions
        if action_type in ["create", "delete"] and requires_letter:
            notification_type = {
                "create": "transaction_created",
                "update": "transaction_updated",
                "delete": "transaction_deleted"
            }.get(action_type)

            # Send the initial notification
            notification = NotificationService.send_transaction_notification(
                transaction=transaction,
                notification_type=notification_type,
                requires_letter=True
            )

            # Set a one-month deadline for the official letter
            deadline = timezone.now() + timezone.timedelta(days=30)

            # Send a specific notification about the official letter requirement
            NotificationService.send_to_role(
                role='president',  # Use 'president' instead of 'admin'
                title="Lettre Officielle Requise pour Don Externe",
                message=f"Une lettre officielle doit être envoyée au Premier Ministère pour le don de {transaction.amount} TND reçu de {getattr(transaction.donor_details, 'name', 'donateur externe')}. Cette lettre doit être envoyée avant le {deadline.strftime('%d/%m/%Y')}.",
                notification_type='official_letter_required',
                related_object=transaction,
                url="/finance",
                priority='high',
                requires_action=True,
                action_deadline=deadline,
                requires_official_letter=True,
                official_letter_recipient="Premier Ministère"
            )

            # Also schedule a reminder 15 days before the deadline
            reminder_date = deadline - timezone.timedelta(days=15)
            # Here, you would typically use a task scheduler like Celery
            # Example: schedule_letter_reminder.apply_async(args=[notification.id], eta=reminder_date)

            # Schedule another reminder 7 days before the deadline
            final_reminder_date = deadline - timezone.timedelta(days=7)
            # Example: schedule_urgent_letter_reminder.apply_async(args=[notification.id], eta=final_reminder_date)

            return notification

        # For ordinary transactions
        notification_type = {
            "create": "transaction_created",
            "update": "transaction_updated",
            "delete": "transaction_deleted"
        }.get(action_type)

        return NotificationService.send_transaction_notification(
            transaction=transaction,
            notification_type=notification_type,
            requires_letter=False
        )

    @staticmethod
    def schedule_monthly_meeting_reminder():
        """
        Schedule a reminder to create monthly meetings
        """
        from datetime import datetime

        now = timezone.now()
        last_day_of_month = (now.replace(day=28) + timezone.timedelta(days=4)).replace(day=1) - timezone.timedelta(
            days=1)
        days_until_end_of_month = (last_day_of_month.date() - now.date()).days

        # If we're approaching the end of the month (less than 7 days)
        if days_until_end_of_month <= 7:
            # Check if we already have a meeting scheduled for next month
            next_month = now + timezone.timedelta(days=days_until_end_of_month + 1)
            next_month_start = next_month.replace(day=1)
            next_month_end = (next_month_start.replace(day=28) + timezone.timedelta(days=4)).replace(
                day=1) - timezone.timedelta(seconds=1)

            # This would normally check the Meeting model, which we're simulating here
            # has_next_month_meeting = Meeting.objects.filter(
            #     start_date__gte=next_month_start,
            #     start_date__lte=next_month_end
            # ).exists()

            # For demonstration purposes, let's assume we don't have a meeting scheduled
            has_next_month_meeting = False

            if not has_next_month_meeting:
                return NotificationService.send_to_role(
                    role='president',  # Use 'president' instead of 'admin'
                    title="Réunion mensuelle à programmer",
                    message=f"Une réunion mensuelle pour {next_month_start.strftime('%B %Y')} doit être programmée. "
                            f"Veuillez créer une nouvelle réunion dès que possible.",
                    notification_type='admin_action_required',
                    url="/meetings/create",
                    priority='high',
                    requires_action=True,
                    action_deadline=last_day_of_month
                )

        return None

    @staticmethod
    def check_monthly_meeting_status():
        """
        Check if meetings are scheduled for the current and upcoming month
        This function should be run periodically (e.g. weekly) via a scheduled task
        """
        from datetime import datetime

        now = timezone.now()
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Calculate the end of the current month
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1)
        else:
            next_month = now.replace(month=now.month + 1)

        current_month_end = next_month.replace(day=1) - timezone.timedelta(seconds=1)

        # Check if we have a meeting for the current month
        # This would normally check the Meeting model, which we're simulating here
        # has_current_month_meeting = Meeting.objects.filter(
        #     start_date__gte=current_month_start,
        #     start_date__lte=current_month_end,
        #     status__in=['scheduled', 'completed']
        # ).exists()

        # For demonstration purposes, let's assume we don't have a meeting scheduled
        has_current_month_meeting = False

        if not has_current_month_meeting:
            # No meeting for the current month - this is a high priority!
            NotificationService.send_to_role(
                role='president',  # Use 'president' instead of 'admin'
                title=f"ATTENTION : Aucune réunion programmée pour {current_month_start.strftime('%B %Y')}",
                message=(
                    f"Il n'y a actuellement aucune réunion programmée pour {current_month_start.strftime('%B %Y')}. "
                    f"Selon les règlements de l'association, au moins une réunion doit être tenue chaque mois. "
                    f"Veuillez programmer une réunion dès que possible."
                ),
                notification_type='admin_action_required',
                url="/meetings/create",
                priority='high',
                requires_action=True,
                action_deadline=current_month_end
            )

        # Also check for next month (reuse the existing function)
        NotificationService.schedule_monthly_meeting_reminder()

        return {
            "current_month": current_month_start.strftime('%B %Y'),
            "has_current_month_meeting": has_current_month_meeting
        }

    @staticmethod
    def send_letter_deadline_reminder(notification_id, urgent=False):
        """
        Send a reminder about an approaching letter deadline

        Args:
            notification_id: ID of the original notification
            urgent: If this is an urgent (final) reminder
        """
        try:
            notification = Notification.objects.get(id=notification_id)

            # Only send if the letter hasn't been sent yet
            if notification.official_letter_sent:
                return None

            days_left = (notification.action_deadline - timezone.now()).days

            if urgent:
                title = "URGENT : Échéance de Lettre Officielle Approchante"
                message = f"L'échéance pour l'envoi de la lettre officielle au Premier Ministère est dans {days_left} jours. Ceci est requis pour le don externe de {notification.related_object.amount} TND reçu le {notification.created_at.strftime('%d/%m/%Y')}."
                priority = 'high'
            else:
                title = "Rappel : Lettre Officielle Bientôt Due"
                message = f"N'oubliez pas d'envoyer la lettre officielle au Premier Ministère dans les {days_left} jours pour le don externe de {notification.related_object.amount} TND."
                priority = 'medium'

            return NotificationService.send_to_role(
                role='president',  # Use 'president' instead of 'admin'
                title=title,
                message=message,
                notification_type='official_letter_required',
                related_object=notification.related_object,
                url="/finance",
                priority=priority,
                requires_action=True,
                action_deadline=notification.action_deadline,
                requires_official_letter=True,
                official_letter_recipient="Premier Ministère"
            )
        except Notification.DoesNotExist:
            return None

    @staticmethod
    def check_pending_reports():
        """
        Check for meetings that need reports
        """
        # This would normally check the Meeting model
        # meetings_without_reports = Meeting.objects.filter(
        #     end_date__lt=timezone.now() - timezone.timedelta(days=3),
        #     report__isnull=True
        # )

        # For demonstration purposes, let's simulate creating notifications
        # for meeting_without_report in meetings_without_reports:
        #     NotificationService.send_report_due_notification(meeting_without_report)

        # Return for demonstration
        return "Checking for pending reports completed"

    @staticmethod
    def check_expiring_letter_deadlines():
        """
        Check for official letters approaching their deadlines
        that haven't been marked as sent
        """
        now = timezone.now()

        # Find notifications that:
        # 1. Require official letters
        # 2. Haven't been marked as sent
        # 3. Have a deadline within the next 7 days
        expiring_letters = Notification.objects.filter(
            requires_official_letter=True,
            official_letter_sent=False,
            action_deadline__lte=now + timezone.timedelta(days=7),
            action_deadline__gt=now
        )

        for letter in expiring_letters:
            days_left = (letter.action_deadline - now).days

            # For very urgent cases (less than 3 days remaining)
            if days_left <= 3:
                NotificationService.send_to_role(
                    role='president',  # Use 'president' instead of 'admin'
                    title=f"CRITIQUE : Lettre Officielle Due dans {days_left} Jours",
                    message=(
                        f"Une lettre officielle au Premier Ministère est due dans {days_left} jours et n'a pas été marquée comme envoyée. "
                        f"Cela concerne le don externe de {letter.related_object.amount} TND reçu le "
                        f"{letter.created_at.strftime('%d/%m/%Y')}. Ne pas envoyer cette lettre peut avoir des implications légales."
                    ),
                    notification_type='official_letter_required',
                    related_object=letter.related_object,
                    url="/finance",
                    priority='high',
                    requires_action=True,
                    action_deadline=letter.action_deadline,
                    requires_official_letter=True,
                    official_letter_recipient="Premier Ministère"
                )

        return len(expiring_letters)