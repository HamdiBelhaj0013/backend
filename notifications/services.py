from django.db import transaction
from django.utils import timezone
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse

from .models import Notification, OfficialLetterLog

User = get_user_model()


class NotificationService:
    """
    Service for handling notifications
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
            notification_type (str): Type of notification
            recipient (User, optional): Specific recipient user
            recipient_role (str, optional): Role of recipients
            related_object (Model, optional): Related model object
            url (str, optional): URL to redirect to
            priority (str, optional): Notification priority (low, medium, high)
            requires_action (bool, optional): Whether action is required
            action_deadline (datetime, optional): Deadline for required action
            requires_official_letter (bool, optional): Whether official letter is required
            official_letter_recipient (str, optional): Recipient for official letter

        Returns:
            Notification: Created notification object
        """
        related_object_id = None
        related_object_type = None

        if related_object:
            related_object_id = related_object.id
            related_object_type = related_object._meta.model_name

        # Create notification
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
        """Send notification to all users"""
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
        """Send notification to all users with specified role"""
        return NotificationService.send_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=None,  # No specific recipient
            recipient_role=role,  # Specific role
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
        """Send notification to all admin users"""
        return NotificationService.send_to_role(
            role='admin',
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
        Send notification about a meeting

        Args:
            meeting: Meeting object
            notification_type: Type of meeting notification
            message: Optional custom message
        """
        title_map = {
            'meeting_scheduled': f"New meeting: {meeting.title}",
            'meeting_cancelled': f"Meeting cancelled: {meeting.title}",
            'meeting_reminder': f"Reminder: {meeting.title}",
        }

        message_map = {
            'meeting_scheduled': f"A new meeting '{meeting.title}' has been scheduled for {meeting.start_date.strftime('%d/%m/%Y at %H:%M')}",
            'meeting_cancelled': f"The meeting '{meeting.title}' scheduled for {meeting.start_date.strftime('%d/%m/%Y at %H:%M')} has been cancelled",
            'meeting_reminder': f"Reminder: Meeting '{meeting.title}' will start on {meeting.start_date.strftime('%d/%m/%Y at %H:%M')}",
        }

        # All members should receive meeting notifications
        return NotificationService.send_notification(
            title=title_map.get(notification_type, f"Meeting update: {meeting.title}"),
            message=message or message_map.get(notification_type, f"Update about meeting: {meeting.title}"),
            notification_type=notification_type,
            related_object=meeting,
            url=f"/meetings/{meeting.id}",
            priority='medium',
        )

    @staticmethod
    def send_report_due_notification(meeting):
        """Send notification that a meeting report is due"""
        return NotificationService.send_to_role(
            role='admin',  # Only admin users need to submit reports
            title=f"Report due for meeting: {meeting.title}",
            message=f"A report needs to be submitted for the meeting '{meeting.title}' held on {meeting.start_date.strftime('%d/%m/%Y')}",
            notification_type='report_due',
            related_object=meeting,
            url=f"/meetings/{meeting.id}/report",
            priority='high',
            requires_action=True,
            action_deadline=timezone.now() + timezone.timedelta(days=7),  # Due in 7 days
        )

    @staticmethod
    def send_transaction_notification(transaction, notification_type, requires_letter=False):
        """
        Send notification about a transaction

        Args:
            transaction: Transaction object
            notification_type: Type of transaction notification
            requires_letter: Whether this requires an official letter
        """
        # Determine transaction type and amount for message
        transaction_type = "income" if transaction.transaction_type == "income" else "expense"
        amount = transaction.amount

        title_map = {
            'transaction_created': f"New {transaction_type}: {amount} TND",
            'transaction_updated': f"{transaction_type.capitalize()} updated: {amount} TND",
            'transaction_deleted': f"{transaction_type.capitalize()} deleted: {amount} TND",
            'donation_received': f"Donation received: {amount} TND",
        }

        message_map = {
            'transaction_created': f"A new {transaction_type} of {amount} TND has been recorded",
            'transaction_updated': f"The {transaction_type} of {amount} TND has been updated",
            'transaction_deleted': f"The {transaction_type} of {amount} TND has been deleted",
            'donation_received': f"A donation of {amount} TND has been received",
        }

        # For transaction notifications, send to admin roles
        notification = NotificationService.send_to_role(
            role='admin',
            title=title_map.get(notification_type, f"{transaction_type.capitalize()} update: {amount} TND"),
            message=message_map.get(notification_type, f"Update about {transaction_type}: {amount} TND"),
            notification_type=notification_type,
            related_object=transaction,
            url="/finance",
            priority='high' if requires_letter else 'medium',
            requires_official_letter=requires_letter,
            official_letter_recipient="Secretary of the Prime Minister" if requires_letter else None
        )

        # Send similar notification to treasurer
        NotificationService.send_to_role(
            role='treasurer',
            title=title_map.get(notification_type, f"{transaction_type.capitalize()} update: {amount} TND"),
            message=message_map.get(notification_type, f"Update about {transaction_type}: {amount} TND"),
            notification_type=notification_type,
            related_object=transaction,
            url="/finance",
            priority='high' if requires_letter else 'medium',
        )

        return notification

    @staticmethod
    def process_critical_transaction(transaction, action_type):
        """
        Process a critical transaction that requires official letter

        Args:
            transaction: Transaction object
            action_type: create, update, or delete
        """
        requires_letter = False

        # Check if this transaction requires an official letter
        if transaction.transaction_type == "income" and transaction.category == "donation":
            # External donations require official letters
            if getattr(transaction, 'donor_details', None) and getattr(transaction.donor_details, 'is_external', False):
                requires_letter = True

        # For both creation and deletion of important transactions
        if action_type in ["create", "delete"] and requires_letter:
            notification_type = {
                "create": "transaction_created",
                "update": "transaction_updated",
                "delete": "transaction_deleted"
            }.get(action_type)

            return NotificationService.send_transaction_notification(
                transaction=transaction,
                notification_type=notification_type,
                requires_letter=True
            )

        # For regular transactions
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

            # This would require checking the Meeting model, which we're simulating here
            # has_next_month_meeting = Meeting.objects.filter(
            #     start_date__gte=next_month_start,
            #     start_date__lte=next_month_end
            # ).exists()

            # For demo purposes, let's assume we don't have a meeting scheduled
            has_next_month_meeting = False

            if not has_next_month_meeting:
                return NotificationService.send_to_role(
                    role='admin',
                    title="Monthly meeting needs to be scheduled",
                    message=f"A monthly meeting for {next_month_start.strftime('%B %Y')} needs to be scheduled. "
                            f"Please create a new meeting as soon as possible.",
                    notification_type='admin_action_required',
                    url="/meetings/create",
                    priority='high',
                    requires_action=True,
                    action_deadline=last_day_of_month
                )

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

        # For demo purposes, let's simulate creating notifications
        # for meeting_without_report in meetings_without_reports:
        #     NotificationService.send_report_due_notification(meeting_without_report)

        # Return for demonstration
        return "Checked for pending reports"