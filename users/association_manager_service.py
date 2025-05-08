"""
Association Manager Service - Isolated service for association presidents to manage users and associations.
"""

from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import PermissionDenied

from users.models import AssociationAccount, Role, CustomUser
from api.models import Member
from users.document_extractor_utils import process_association_verification

from datetime import date

User = get_user_model()


class AssociationManagerService:
    """
    Service for association presidents to manage users and associations.
    Provides isolated business logic for user and association management.
    """

    def __init__(self, user):
        """
        Initialize the service with the requesting user

        Args:
            user: The authenticated user making the request
        """
        self.user = user
        self._validate_president_role()
        self.association = user.association

    def _validate_president_role(self):
        """
        Validate that the user has the president role or is a superuser

        Raises:
            PermissionDenied: If the user is not a president or superuser
        """
        # Check if superuser (has full access)
        if self.user.is_superuser:
            return True

        # Check if user has a role
        if not hasattr(self.user, 'role') or not self.user.role:
            raise PermissionDenied("You don't have permission to manage the association.")

        # Check if user has president role
        if self.user.role.name != 'president':
            raise PermissionDenied("Only association presidents can manage the association.")

        # Check if user belongs to an association
        if not self.user.association:
            raise PermissionDenied("You are not associated with any organization.")

        return True

    def _validate_same_association(self, target_user):
        """
        Validate that the target user belongs to the same association as the requesting user

        Args:
            target_user: The user to check

        Raises:
            PermissionDenied: If the target user belongs to a different association
        """
        if not target_user.association or target_user.association.id != self.association.id:
            raise PermissionDenied("You can only manage users in your own association.")

    def get_association_details(self):
        """
        Get details about the association

        Returns:
            AssociationAccount: The association object
        """
        if not self.association:
            raise PermissionDenied("You don't have an associated organization.")

        return self.association

    def update_association_details(self, data):
        """
        Update the association details

        Args:
            data: Dictionary containing the fields to update

        Returns:
            AssociationAccount: The updated association object
        """
        if not self.association:
            raise PermissionDenied("You don't have an associated organization.")

        # List of fields that can be updated by the president
        allowed_fields = [
            'name', 'email',
            'president_name', 'president_email',
            'treasurer_name', 'treasurer_email',
            'secretary_name', 'secretary_email',
            'cin_recto', 'cin_verso',
            'matricule_fiscal', 'rne_document'
        ]

        # Only update allowed fields
        for field in allowed_fields:
            if field in data:
                setattr(self.association, field, data[field])

        # Save the updated association
        self.association.save()

        # If RNE document or matricule_fiscal changed, attempt verification
        rne_changed = 'rne_document' in data
        matricule_changed = 'matricule_fiscal' in data

        if (rne_changed or matricule_changed) and self.association.rne_document and self.association.matricule_fiscal:
            # Attempt to verify the association
            return process_association_verification(self.association)

        return self.association

    def get_users(self, filters=None):
        """
        Get all users in the association with optional filtering

        Args:
            filters: Optional dictionary of filters to apply

        Returns:
            QuerySet: Filtered users in the association
        """
        if not self.association:
            return User.objects.none()

        # Start with users in the association
        users = User.objects.filter(association=self.association)

        # Apply filters if provided
        if filters:
            # Filter by role
            if 'role' in filters:
                users = users.filter(role__name=filters['role'])

            # Filter by validation status
            if 'is_validated' in filters:
                users = users.filter(is_validated=filters['is_validated'])

            # Filter by email
            if 'email' in filters:
                users = users.filter(email__icontains=filters['email'])

            # Filter by name
            if 'full_name' in filters:
                users = users.filter(full_name__icontains=filters['full_name'])

        return users

    def get_user(self, user_id):
        """
        Get a specific user in the association

        Args:
            user_id: ID of the user to get

        Returns:
            CustomUser: The user object

        Raises:
            PermissionDenied: If the user is not in the same association
        """
        user = User.objects.get(pk=user_id)
        self._validate_same_association(user)
        return user

    def update_user(self, user_id, data):
        """
        Update a user's details

        Args:
            user_id: ID of the user to update
            data: Dictionary of data to update

        Returns:
            CustomUser: The updated user
        """
        user = self.get_user(user_id)

        # Fields that presidents can update for other users
        allowed_fields = ['full_name', 'role', 'is_validated']

        # Only update allowed fields
        for field in allowed_fields:
            if field in data:
                # Special handling for role
                if field == 'role':
                    if isinstance(data[field], int):
                        # Role ID provided
                        role = Role.objects.get(pk=data[field])
                        user.role = role
                    elif isinstance(data[field], dict) and 'id' in data[field]:
                        # Role object with ID provided
                        role = Role.objects.get(pk=data[field]['id'])
                        user.role = role
                else:
                    setattr(user, field, data[field])

        # If validating the user for the first time
        if 'is_validated' in data and data['is_validated'] and not user.is_validated:
            user.validated_by = self.user
            user.validation_date = timezone.now()

            # Ensure a Member record exists for this user
            self._ensure_member_record_exists(user)

        user.save()
        return user

    def validate_user(self, user_id, validate=True):
        """
        Validate or reject a user

        Args:
            user_id: ID of the user to validate
            validate: True to validate, False to reject

        Returns:
            CustomUser: The updated user
        """
        user = self.get_user(user_id)

        if validate:
            # Validate user
            user.is_validated = True
            user.validated_by = self.user
            user.validation_date = timezone.now()
            user.save()

            # Ensure a Member record exists
            self._ensure_member_record_exists(user)

            return user
        else:
            # Reject user
            user.is_validated = False
            user.save()

            # Regular members can be deleted when rejected
            if not user.role or user.role.name == 'member':
                # First try to delete any Member record
                try:
                    member = Member.objects.get(email=user.email)
                    member.delete()
                except Member.DoesNotExist:
                    pass

                # Then delete the user
                user.delete()
                return None

            return user

    def _ensure_member_record_exists(self, user):
        """
        Ensure a Member record exists for a user

        Args:
            user: The user to create a Member record for

        Returns:
            Member: The new or existing Member record
        """
        try:
            # Check if a Member record already exists
            member = Member.objects.get(email=user.email)

            # If member exists but has different association, update it
            if member.association != user.association:
                member.association = user.association
                member.save()

            return member
        except Member.DoesNotExist:
            # Create a new Member record
            today = date.today()

            # Create the member
            new_member = Member(
                email=user.email,
                name=user.full_name or user.email,
                cin=user.cin,
                birth_date=user.birth_date or date(2000, 1, 1),  # Default birth date if not available
                address="Please update your address",  # Placeholder
                nationality="Please update your nationality",  # Placeholder
                job="Please update your job",  # Placeholder
                joining_date=today,
                role="Membre",  # Default role for regular members
                association=user.association,
                needs_profile_completion=True  # Always true for new members
            )

            # Save with skip_validation to bypass potential validation issues
            new_member.save(skip_validation=True)
            return new_member

    def create_user(self, data):
        """
        Create a new user in the association

        Args:
            data: Dictionary containing user details

        Returns:
            CustomUser: The newly created user
        """
        with transaction.atomic():
            email = data.get('email')
            if not email:
                raise ValueError("Email is required")

            # Check if user already exists
            if User.objects.filter(email=email).exists():
                raise ValueError(f"User with email {email} already exists")

            # Set up required fields
            user_data = {
                'email': email,
                'full_name': data.get('full_name', ''),
                'association': self.association,
                'is_validated': data.get('is_validated', False),
            }

            # Set role if provided, default to 'member'
            role_id = data.get('role_id')
            if role_id:
                try:
                    role = Role.objects.get(pk=role_id)
                    user_data['role'] = role
                except Role.DoesNotExist:
                    # Default to member role
                    role = Role.objects.get(name='member')
                    user_data['role'] = role
            else:
                # Default to member role
                role = Role.objects.get(name='member')
                user_data['role'] = role

            # Set CIN if provided
            if 'cin' in data:
                user_data['cin'] = data['cin']

            # Set birth_date if provided
            if 'birth_date' in data:
                user_data['birth_date'] = data['birth_date']

            # Create the user
            password = data.get('password')
            if not password:
                # Generate a temporary password if not provided
                import secrets
                password = secrets.token_urlsafe(12)

            # Create user
            user = User.objects.create_user(**user_data)
            user.set_password(password)

            # If user is validated, set validation details
            if user_data['is_validated']:
                user.validated_by = self.user
                user.validation_date = timezone.now()

            user.save()

            # If user is validated, ensure a Member record exists
            if user_data['is_validated']:
                self._ensure_member_record_exists(user)

            return user, password

    def delete_user(self, user_id):
        """
        Delete a user from the association

        Args:
            user_id: ID of the user to delete

        Returns:
            bool: True if successful

        Raises:
            PermissionDenied: If trying to delete a protected role or the requesting user
        """
        user = self.get_user(user_id)

        # Prevent deleting yourself
        if user.id == self.user.id:
            raise PermissionDenied("You cannot delete your own account")

        # Prevent deleting users with protected roles
        protected_roles = ['president', 'treasurer', 'secretary']
        if user.role and user.role.name in protected_roles:
            raise PermissionDenied(f"Cannot delete user with {user.role.name} role")

        # Delete any Member record
        try:
            member = Member.objects.get(email=user.email)
            member.delete()
        except Member.DoesNotExist:
            pass

        # Delete the user
        user.delete()
        return True

    def set_user_role(self, user_id, role_id):
        """
        Set a user's role

        Args:
            user_id: ID of the user to update
            role_id: ID of the role to assign

        Returns:
            CustomUser: The updated user
        """
        user = self.get_user(user_id)

        # Get the role
        try:
            role = Role.objects.get(pk=role_id)
        except Role.DoesNotExist:
            raise ValueError(f"Role with ID {role_id} does not exist")

        # Check if another user already has this role (for unique roles)
        unique_roles = ['president', 'treasurer', 'secretary']
        if role.name in unique_roles:
            existing_user = User.objects.filter(
                association=self.association,
                role=role
            ).exclude(pk=user_id).first()

            if existing_user:
                raise ValueError(f"Another user already has the {role.name} role")

        # Set the user's role
        user.role = role
        user.save()

        # If the role corresponds to a specific role in the Member record, update it
        try:
            member = Member.objects.get(email=user.email)

            # Map role names to Member role values
            role_mapping = {
                'president': 'Président',
                'treasurer': 'Trésorier',
                'secretary': 'Secrétaire générale',
                'member': 'Membre'
            }

            if role.name in role_mapping:
                member.role = role_mapping[role.name]
                member.save()
        except Member.DoesNotExist:
            # If no member record exists and user is validated, create one
            if user.is_validated:
                self._ensure_member_record_exists(user)

        return user

    def send_password_reset(self, user_id):
        """
        Send a password reset email to a user

        Args:
            user_id: ID of the user to send the reset email to

        Returns:
            bool: True if successful
        """
        from django_rest_passwordreset.models import ResetPasswordToken
        from django.template.loader import render_to_string
        from django.core.mail import EmailMultiAlternatives
        from django.utils.html import strip_tags

        user = self.get_user(user_id)

        # Create password reset token
        token = ResetPasswordToken.objects.create(
            user=user,
            user_agent="API",
            ip_address="127.0.0.1"
        )

        # Generate reset link
        sitelink = "http://localhost:5173/"
        token_key = "{}".format(token.key)
        full_link = str(sitelink) + str("password-reset/") + str(token_key)

        context = {
            'full_link': full_link,
            'email_adress': user.email,
            'full_name': user.full_name or user.email
        }

        html_message = render_to_string("backend/email.html", context=context)
        plain_message = strip_tags(html_message)

        msg = EmailMultiAlternatives(
            subject="Password reset for {title}".format(title=user.email),
            body=plain_message,
            from_email="noreply.myorg@gmail.com",
            to=[user.email]
        )

        msg.attach_alternative(html_message, "text/html")
        msg.send()

        return True

    def get_roles(self):
        """
        Get all available roles

        Returns:
            QuerySet: All Role objects
        """
        return Role.objects.all()

    def get_members(self, filters=None):
        """
        Get all members in the association with optional filtering

        Args:
            filters: Optional dictionary of filters to apply

        Returns:
            QuerySet: Filtered members in the association
        """
        if not self.association:
            return Member.objects.none()

        # Start with members in the association
        members = Member.objects.filter(association=self.association)

        # Apply filters if provided
        if filters:
            # Filter by role
            if 'role' in filters:
                members = members.filter(role__icontains=filters['role'])

            # Filter by name
            if 'name' in filters:
                members = members.filter(name__icontains=filters['name'])

            # Filter by email
            if 'email' in filters:
                members = members.filter(email__icontains=filters['email'])

            # Filter by needs_profile_completion
            if 'needs_profile_completion' in filters:
                members = members.filter(needs_profile_completion=filters['needs_profile_completion'])

        return members