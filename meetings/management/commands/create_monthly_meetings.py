import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from meetings.models import Meeting
from users.models import AssociationAccount
from dateutil.relativedelta import relativedelta


class Command(BaseCommand):
    help = 'Create monthly meetings for associations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--months',
            type=int,
            default=3,
            help='Number of months to create meetings for (default: 3)'
        )
        parser.add_argument(
            '--day',
            type=int,
            default=15,
            help='Day of month for meetings (default: 15)'
        )
        parser.add_argument(
            '--hour',
            type=int,
            default=18,
            help='Hour for meetings (24-hour format, default: 18/6PM)'
        )
        parser.add_argument(
            '--duration',
            type=int,
            default=2,
            help='Meeting duration in hours (default: 2)'
        )
        parser.add_argument(
            '--association_id',
            type=int,
            help='Specific association ID to create meetings for (optional)'
        )

    def handle(self, *args, **options):
        months = options['months']
        day_of_month = options['day']
        hour = options['hour']
        duration = options['duration']
        association_id = options.get('association_id')

        # Validate inputs
        if months <= 0 or months > 12:
            self.stdout.write(self.style.ERROR('Months must be between 1 and 12'))
            return

        if day_of_month < 1 or day_of_month > 28:
            self.stdout.write(self.style.ERROR('Day must be between 1 and 28 to ensure valid dates for all months'))
            return

        if hour < 0 or hour > 23:
            self.stdout.write(self.style.ERROR('Hour must be between 0 and 23'))
            return

        if duration <= 0 or duration > 8:
            self.stdout.write(self.style.ERROR('Duration must be between 1 and 8 hours'))
            return

        # Get associations
        if association_id:
            associations = AssociationAccount.objects.filter(id=association_id)
            if not associations.exists():
                self.stdout.write(self.style.ERROR(f'Association with ID {association_id} not found'))
                return
        else:
            associations = AssociationAccount.objects.all()

        total_created = 0
        now = timezone.now()

        # For each association
        for association in associations:
            # Get any existing meetings in the next months to avoid duplicates
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = (start_date + relativedelta(months=months)).replace(day=28)

            existing_meetings = Meeting.objects.filter(
                association=association,
                meeting_type='regular',
                start_date__gte=start_date,
                start_date__lte=end_date
            )

            # Collect existing meeting months to avoid duplicates
            existing_months = set()
            for meeting in existing_meetings:
                meeting_key = f"{meeting.start_date.year}-{meeting.start_date.month}"
                existing_months.add(meeting_key)

            # Create meetings for each month if they don't exist
            meetings_created = 0

            for i in range(months):
                # Calculate meeting date
                meeting_date = now + relativedelta(months=i)
                meeting_date = meeting_date.replace(day=day_of_month, hour=hour, minute=0, second=0, microsecond=0)

                # Skip if this month already has a meeting
                meeting_key = f"{meeting_date.year}-{meeting_date.month}"
                if meeting_key in existing_months:
                    self.stdout.write(
                        f"Skipping {meeting_date.strftime('%B %Y')} for {association.name} - meeting already exists"
                    )
                    continue

                # Skip dates in the past
                if meeting_date < now:
                    self.stdout.write(
                        f"Skipping {meeting_date.strftime('%B %Y')} for {association.name} - date is in the past"
                    )
                    continue

                # Create the meeting
                meeting = Meeting(
                    title=f"Monthly Meeting - {meeting_date.strftime('%B %Y')}",
                    description=f"Regular monthly meeting for {association.name}",
                    meeting_type='regular',
                    status='scheduled',
                    association=association,
                    start_date=meeting_date,
                    end_date=meeting_date + datetime.timedelta(hours=duration),
                    location=f"{association.name} Office",
                    agenda=(
                        "1. Welcome and Introduction\n"
                        "2. Approval of Previous Meeting Minutes\n"
                        "3. Financial Update\n"
                        "4. Project Updates\n"
                        "5. New Business\n"
                        "6. Open Discussion\n"
                        "7. Action Items Review\n"
                        "8. Next Meeting Date\n"
                    ),
                    is_recurring=True,
                    recurrence_pattern={
                        'frequency': 'monthly',
                        'interval': 1,
                        'day_of_month': day_of_month
                    },
                    notification_method='both',
                    reminder_days_before=2
                )
                meeting.save()

                meetings_created += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Created meeting for {association.name} on {meeting_date.strftime('%B %d, %Y at %I:%M %p')}"
                    )
                )

            self.stdout.write(f"Created {meetings_created} meetings for {association.name}")
            total_created += meetings_created

        self.stdout.write(self.style.SUCCESS(
            f"Successfully created {total_created} meetings across {associations.count()} associations"))