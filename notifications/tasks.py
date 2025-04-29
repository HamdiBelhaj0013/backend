from datetime import timedelta
from django.utils import timezone
from django.apps import apps
from django.db.models import Q, Sum, F

from .services import NotificationService


# If using Celery, you would import the following:
# from celery import shared_task


# @shared_task
def check_monthly_meetings():
    """
    Check if monthly meetings are scheduled
    - Run daily
    """
    return NotificationService.schedule_monthly_meeting_reminder()


# @shared_task
def check_pending_reports():
    """
    Check for meetings that need reports
    - Run daily
    """
    Meeting = apps.get_model('meetings', 'Meeting')
    Report = apps.get_model('meetings', 'Report', require_ready=False)

    # Find meetings that ended more than 3 days ago with no reports
    cutoff_date = timezone.now() - timedelta(days=3)

    meetings_without_reports = Meeting.objects.filter(
        end_date__lt=cutoff_date,
        status='completed'
    ).exclude(
        id__in=Report.objects.values_list('meeting_id', flat=True)
    )

    for meeting in meetings_without_reports:
        NotificationService.send_report_due_notification(meeting)

    return f"Checked {meetings_without_reports.count()} meetings requiring reports"


# @shared_task
def check_upcoming_meetings():
    """
    Check for upcoming meetings and send reminders
    - Run daily
    """
    Meeting = apps.get_model('meetings', 'Meeting')

    # Find meetings happening tomorrow
    tomorrow_start = timezone.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
    tomorrow_end = tomorrow_start.replace(hour=23, minute=59, second=59)

    upcoming_meetings = Meeting.objects.filter(
        start_date__gte=tomorrow_start,
        start_date__lte=tomorrow_end,
        status='scheduled'
    )

    for meeting in upcoming_meetings:
        NotificationService.send_meeting_notification(
            meeting=meeting,
            notification_type="meeting_reminder"
        )

    return f"Sent reminders for {upcoming_meetings.count()} upcoming meetings"


# @shared_task
def check_budget_thresholds():
    """
    Check if project budgets are approaching their limits
    - Run weekly
    """
    Transaction = apps.get_model('finances', 'Transaction')
    Project = apps.get_model('projects', 'Project')
    BudgetAllocation = apps.get_model('finances', 'BudgetAllocation')

    # Find all active project budgets
    active_budgets = BudgetAllocation.objects.filter(
        project__status='active'
    )

    for budget in active_budgets:
        # Calculate how much has been spent
        spent = Transaction.objects.filter(
            budget_allocation=budget,
            transaction_type='expense',
            status='verified'
        ).aggregate(total=Sum('amount'))['total'] or 0

        # Calculate percentage spent
        if budget.allocated_amount > 0:  # Avoid division by zero
            percentage_spent = (spent / budget.allocated_amount) * 100

            # If spent more than 80% of budget, send notification
            if percentage_spent >= 80 and percentage_spent < 100:
                NotificationService.send_to_role(
                    role='treasurer',
                    title=f"Budget threshold alert: {budget.project.name}",
                    message=(
                        f"The budget for project '{budget.project.name}' is at {percentage_spent:.1f}% "
                        f"({spent} of {budget.allocated_amount}). Please review and take action if needed."
                    ),
                    notification_type="budget_threshold",
                    related_object=budget.project,
                    url=f"/projects/{budget.project.id}",
                    priority='medium'
                )

            # If over budget, send high priority notification
            elif percentage_spent >= 100:
                NotificationService.send_to_role(
                    role='treasurer',
                    title=f"Budget exceeded: {budget.project.name}",
                    message=(
                        f"The budget for project '{budget.project.name}' has been exceeded! "
                        f"Currently at {percentage_spent:.1f}% ({spent} of {budget.allocated_amount}). "
                        f"Immediate action required."
                    ),
                    notification_type="budget_threshold",
                    related_object=budget.project,
                    url=f"/projects/{budget.project.id}",
                    priority='high',
                    requires_action=True
                )

    return f"Checked {active_budgets.count()} project budgets"


# @shared_task
def check_pending_official_letters():
    """
    Check for notifications requiring official letters that haven't been sent
    - Run weekly
    """
    from .models import Notification

    # Find notifications requiring letters not yet sent
    pending_letters = Notification.objects.filter(
        requires_official_letter=True,
        official_letter_sent=False,
        created_at__lt=timezone.now() - timedelta(days=3)  # Give a few days buffer
    )

    if pending_letters.exists():
        # Send reminder to admins
        NotificationService.send_to_role(
            role='admin',
            title="Pending official letters require attention",
            message=(
                f"There are {pending_letters.count()} pending official letters that need to be sent. "
                f"Please review and take action as soon as possible."
            ),
            notification_type="admin_action_required",
            url="/notifications?tab=3",  # Go to Official Letters tab
            priority='high',
            requires_action=True
        )

    return f"Checked {pending_letters.count()} pending official letters"


# @shared_task
def generate_monthly_summary():
    """
    Generate monthly financial and activity summary
    - Run on the 1st of every month
    """
    # Only run on the first day of the month
    today = timezone.now().date()
    if today.day != 1:
        return "Not first day of month, skipping"

    # Get models
    Transaction = apps.get_model('finances', 'Transaction')
    Meeting = apps.get_model('meetings', 'Meeting')

    # Calculate date ranges for previous month
    first_day_previous_month = today.replace(day=1) - timedelta(days=1)
    first_day_previous_month = first_day_previous_month.replace(day=1)
    last_day_previous_month = today.replace(day=1) - timedelta(days=1)

    # Get financial data for previous month
    monthly_transactions = Transaction.objects.filter(
        date__gte=first_day_previous_month,
        date__lte=last_day_previous_month,
        status='verified'
    )

    total_income = monthly_transactions.filter(
        transaction_type='income'
    ).aggregate(total=Sum('amount'))['total'] or 0

    total_expenses = monthly_transactions.filter(
        transaction_type='expense'
    ).aggregate(total=Sum('amount'))['total'] or 0

    net_balance = total_income - total_expenses

    # Count meetings
    monthly_meetings = Meeting.objects.filter(
        start_date__gte=first_day_previous_month,
        start_date__lte=last_day_previous_month
    )

    meetings_held = monthly_meetings.filter(status='completed').count()
    meetings_cancelled = monthly_meetings.filter(status='cancelled').count()

    # Generate summary message
    month_name = first_day_previous_month.strftime('%B %Y')

    message = (
        f"## Monthly Summary for {month_name}\n\n"
        f"### Financial Overview\n"
        f"- Total Income: {total_income} TND\n"
        f"- Total Expenses: {total_expenses} TND\n"
        f"- Net Balance: {net_balance} TND\n\n"
        f"### Activity Summary\n"
        f"- Meetings Held: {meetings_held}\n"
        f"- Meetings Cancelled: {meetings_cancelled}\n"
    )

    # Send notification to all admins and treasurer
    NotificationService.send_to_role(
        role='admin',
        title=f"Monthly Summary: {month_name}",
        message=message,
        notification_type="monthly_summary",
        url="/finance",
        priority='medium'
    )

    NotificationService.send_to_role(
        role='treasurer',
        title=f"Monthly Financial Summary: {month_name}",
        message=message,
        notification_type="monthly_summary",
        url="/finance",
        priority='medium'
    )

    return f"Generated monthly summary for {month_name}"


# Schedule for Celery
# If using Celery, add these scheduled tasks to your Celery beat schedule:
"""
CELERY_BEAT_SCHEDULE = {
    'check-monthly-meetings': {
        'task': 'notifications.tasks.check_monthly_meetings',
        'schedule': crontab(hour=8),  # Run daily at 8 AM
    },
    'check-pending-reports': {
        'task': 'notifications.tasks.check_pending_reports',
        'schedule': crontab(hour=9),  # Run daily at 9 AM
    },
    'check-upcoming-meetings': {
        'task': 'notifications.tasks.check_upcoming_meetings',
        'schedule': crontab(hour=10),  # Run daily at 10 AM
    },
    'check-budget-thresholds': {
        'task': 'notifications.tasks.check_budget_thresholds',
        'schedule': crontab(day_of_week='monday', hour=8),  # Run weekly on Monday at 8 AM
    },
    'check-pending-official-letters': {
        'task': 'notifications.tasks.check_pending_official_letters',
        'schedule': crontab(day_of_week='monday,thursday', hour=9),  # Run twice a week
    },
    'generate-monthly-summary': {
        'task': 'notifications.tasks.generate_monthly_summary',
        'schedule': crontab(day_of_month='1', hour=7),  # Run on 1st of every month at 7 AM
    },
}
"""