from django.core.management.base import BaseCommand
from users.models import Role


class Command(BaseCommand):
    help = 'Create default roles'

    def handle(self, *args, **options):
        # Define the default roles
        default_roles = [
            ('president', 'Association president with full access'),
            ('treasurer', 'Financial manager with finance access'),
            ('secretary', 'General secretary with administrative access'),
            ('member', 'Regular member with limited access'),
        ]

        # Create each role if it doesn't exist
        for role_name, description in default_roles:
            role, created = Role.objects.get_or_create(
                name=role_name,
                defaults={'description': description}
            )

            if created:
                self.stdout.write(self.style.SUCCESS(f'Created role: {role_name}'))
            else:
                self.stdout.write(f'Role already exists: {role_name}')

        # Count roles to verify
        role_count = Role.objects.count()
        self.stdout.write(self.style.SUCCESS(f'Total roles in database: {role_count}'))