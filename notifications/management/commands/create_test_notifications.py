from django.core.management.base import BaseCommand
from notifications.services import NotificationService
from django.utils import timezone


class Command(BaseCommand):
    help = 'Create test notifications to verify notification system'

    def handle(self, *args, **options):
        self.stdout.write("Creating test notifications...")

        # Create different types of notifications

        # 1. Notification for all users
        notification = NotificationService.send_to_all(
            title="System Announcement",
            message="This is a test announcement for all users in the system.",
            notification_type="admin_action_required",
            url="/home",
            priority="medium"
        )
        self.stdout.write(self.style.SUCCESS(f"Created notification for all users: {notification.id}"))

        # 2. Notification for admins
        notification = NotificationService.send_to_role(
            role='admin',
            title="Admin Alert: Action Required",
            message="This is a high priority notification that requires admin attention.",
            notification_type="admin_action_required",
            url="/home",
            priority="high",
            requires_action=True,
            action_deadline=timezone.now() + timezone.timedelta(days=2)
        )
        self.stdout.write(self.style.SUCCESS(f"Created admin notification: {notification.id}"))

        # 3. Meeting notification
        notification = NotificationService.send_notification(
            title="Meeting Reminder",
            message="Important meeting coming up tomorrow at 10:00 AM.",
            notification_type="meeting_reminder",
            url="/meetings",
            priority="medium",
        )
        self.stdout.write(self.style.SUCCESS(f"Created meeting notification: {notification.id}"))

        # 4. Transaction notification requiring official letter
        notification = NotificationService.send_notification(
            title="Donation Received: 5000 TND",
            message="A donation of 5000 TND has been received from an external donor.",
            notification_type="donation_received",
            url="/finance",
            priority="high",
            requires_official_letter=True,
            official_letter_recipient="Secretary of the Prime Minister"
        )
        self.stdout.write(self.style.SUCCESS(f"Created donation notification: {notification.id}"))

        # 5. Budget threshold notification
        notification = NotificationService.send_to_role(
            role='treasurer',
            title="Budget Alert: Project Alpha",
            message="Project Alpha has reached 85% of its allocated budget.",
            notification_type="budget_threshold",
            url="/finance",
            priority="medium"
        )
        self.stdout.write(self.style.SUCCESS(f"Created budget threshold notification: {notification.id}"))

        # 6. Report due notification
        notification = NotificationService.send_notification(
            title="Report Due: Annual Meeting",
            message="The report for the Annual Meeting is due in 3 days.",
            notification_type="report_due",
            url="/meetings/reports",
            priority="high",
            requires_action=True,
            action_deadline=timezone.now() + timezone.timedelta(days=3)
        )
        self.stdout.write(self.style.SUCCESS(f"Created report due notification: {notification.id}"))

        total_count = 6
        self.stdout.write(self.style.SUCCESS(f"Successfully created {total_count} test notifications"))