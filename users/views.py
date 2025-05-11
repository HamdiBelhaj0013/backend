from django.shortcuts import get_object_or_404
from rest_framework import viewsets, permissions
from django.contrib.auth import get_user_model, authenticate
from knox.models import AuthToken
from .serializers import *
from .serializers import UserProfileSerializer
from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.utils import timezone
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import AssociationAccount, Role, CustomUser
from .serializers import AssociationAccountSerializer, AssociationVerificationSerializer
from .document_extractor_utils import verify_association_document
import re
from datetime import datetime, date
class AssociationAccountViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing association accounts with improved access controls
    """
    serializer_class = AssociationAccountSerializer

    def get_queryset(self):
        """
        Filter queryset based on user's permissions and association
        """
        user = self.request.user

        # Admin users can see all associations
        if user.is_superuser or user.is_staff:
            return AssociationAccount.objects.all()

        # Anonymous users (for verification endpoint)
        if user.is_anonymous:
            return AssociationAccount.objects.all()  # Or filter as needed for anonymous access

        # Regular users can only see their own association
        if hasattr(user, 'association') and user.association:
            return AssociationAccount.objects.filter(id=user.association.id)

        # Users without association don't see anything
        return AssociationAccount.objects.none()

    def get_permissions(self):
        """
        Custom permissions:
        - Admin users can access all endpoints
        - Regular users can only view their own account
        - Allow anonymous access for retrieve (GET) to check verification status
        """
        # Allow anonymous access for retrieve (GET) operation
        if self.action == 'retrieve':
            return [permissions.AllowAny()]

        # Admin-only actions
        if self.action in ['list', 'verify', 'manual_verify']:
            return [IsAdminUser()]

        return [IsAuthenticated()]

    def retrieve(self, request, *args, **kwargs):
        """
        Override retrieve to check if user can access this association
        """
        instance = self.get_object()

        # Non-admin users can only view their own association
        if not (request.user.is_superuser or request.user.is_staff) and not request.user.is_anonymous:
            if hasattr(request.user,
                       'association') and request.user.association and request.user.association.id != instance.id:
                return Response(
                    {"error": "You don't have permission to access this association"},
                    status=403
                )

        serializer = self.get_serializer(instance)
        return Response(serializer.data)


    def perform_create(self, serializer):
        """Override create to attempt automatic verification"""
        association = serializer.save()

        # Check if we can perform automatic verification
        if association.rne_document and association.matricule_fiscal:
            verified_association = self._verify_association(association)

            # If verification was successful, create user accounts
            if verified_association.is_verified:
                self._create_role_based_users(verified_association)

    def _verify_association(self, association):
        """
        Internal method to verify an association's documents
        """
        from .document_extractor_utils import process_association_verification
        return process_association_verification(association)

    # Update the _create_role_based_users method in views.py (AssociationAccountViewSet)

    def _create_role_based_users(self, association):
        """
        Create user accounts for president, treasurer, and secretary
        with appropriate roles using their full names, and create corresponding Member objects.
        These accounts are automatically verified.
        """
        User = get_user_model()
        print(f"Starting user creation for association: {association.name}")

        # Get the Role objects (creating them if they don't exist)
        president_role, _ = Role.objects.get_or_create(name='president')
        treasurer_role, _ = Role.objects.get_or_create(name='treasurer')
        secretary_role, _ = Role.objects.get_or_create(name='secretary')

        # Generate temporary random password
        import secrets
        from datetime import date
        import datetime
        temp_password = secrets.token_urlsafe(12)
        print(f"Generated temporary password for new users")

        # Create users with appropriate roles if emails provided
        created_users = []

        # Import Member model from members app
        from api.models import Member

        today = date.today()
        default_birth_date = datetime.date(2000, 1, 1)

        # List to store created member IDs for notification
        created_member_ids = []

        # Current time for validation timestamp
        current_time = timezone.now()

        if association.president_email:
            # Use provided name or generate default name
            president_name = association.president_name or f"President of {association.name}"
            print(f"Creating president user: {president_name} ({association.president_email})")

            # Create CustomUser first
            president_user, created = User.objects.get_or_create(
                email=association.president_email,
                defaults={
                    'association': association,
                    'full_name': president_name,
                    'role': president_role,
                    'is_validated': True,  # Set to True by default
                    'validation_date': current_time  # Set validation date
                }
            )

            # Only set password if this is a new user
            if created:
                print(f"New president user created. Setting password.")
                president_user.set_password(temp_password)
                president_user.save()
                created_users.append(president_user)
                print(f"Added president to list of created users")
            elif not president_user.is_validated:
                # If user exists but wasn't validated, validate them now
                president_user.is_validated = True
                president_user.validation_date = current_time
                president_user.save()
                print(f"Existing president user validated")

            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.president_email,
                    defaults={
                        'name': president_name,
                        'address': "Modifier Votre adresse",
                        'nationality': "Mettre a jour Votre nationalité",
                        'birth_date': default_birth_date,
                        'job': "Association President",
                        'joining_date': today,
                        'role': "Président",
                        'association': association
                    }
                )

                # Add a flag to indicate this member needs profile completion
                if member_created:
                    member.needs_profile_completion = True
                    member.save()
                    created_member_ids.append(member.id)
                    print(f"Created new Member entry for president: {president_name}")
                else:
                    # Update member with association president role if it already exists
                    if member.association != association or member.role != "Président":
                        member.association = association
                        member.role = "Président"
                        member.save()
                        print(f"Updated existing Member entry for president: {president_name}")
            except Exception as e:
                print(f"Error creating member for president: {str(e)}")
                import traceback
                traceback.print_exc()

        # Similar updates for treasurer with automatic verification
        if association.treasurer_email:
            # Use provided name or generate default name
            treasurer_name = association.treasurer_name or f"Treasurer of {association.name}"
            print(f"Creating treasurer user: {treasurer_name} ({association.treasurer_email})")

            # Create CustomUser first
            treasurer_user, created = User.objects.get_or_create(
                email=association.treasurer_email,
                defaults={
                    'association': association,
                    'full_name': treasurer_name,
                    'role': treasurer_role,
                    'is_validated': True,
                    'validation_date': current_time
                }
            )

            if created:
                print(f"New treasurer user created. Setting password.")
                treasurer_user.set_password(temp_password)
                treasurer_user.save()
                created_users.append(treasurer_user)
                print(f"Added treasurer to list of created users")
            elif not treasurer_user.is_validated:
                # If user exists but wasn't validated, validate them now
                treasurer_user.is_validated = True
                treasurer_user.validation_date = current_time
                treasurer_user.save()
                print(f"Existing treasurer user validated")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.treasurer_email,
                    defaults={
                        'name': treasurer_name,
                        'address': "Mettre a jour votre addresse",
                        'nationality': "Mettre a jour votre nationalité",
                        'birth_date': default_birth_date,
                        'job': "Association Treasurer",
                        'joining_date': today,
                        'role': "Trésorier",
                        'association': association
                    }
                )

                if member_created:
                    member.needs_profile_completion = True
                    member.save()
                    created_member_ids.append(member.id)
                    print(f"Created new Member entry for treasurer: {treasurer_name}")
                else:
                    # Update member with association treasurer role if it already exists
                    if member.association != association or member.role != "Trésorier":
                        member.association = association
                        member.role = "Trésorier"
                        member.save()
                        print(f"Updated existing Member entry for treasurer: {treasurer_name}")
            except Exception as e:
                print(f"Error creating member for treasurer: {str(e)}")
                import traceback
                traceback.print_exc()

        # Similar updates for secretary with automatic verification
        if association.secretary_email:
            # Use provided name or generate default name
            secretary_name = association.secretary_name or f"Secretary of {association.name}"
            print(f"Creating secretary user: {secretary_name} ({association.secretary_email})")

            # Create CustomUser first
            secretary_user, created = User.objects.get_or_create(
                email=association.secretary_email,
                defaults={
                    'association': association,
                    'full_name': secretary_name,
                    'role': secretary_role,
                    'is_validated': True,  # Set to True by default
                    'validation_date': current_time  # Set validation date
                }
            )

            if created:
                print(f"New secretary user created. Setting password.")
                secretary_user.set_password(temp_password)
                secretary_user.save()
                created_users.append(secretary_user)
                print(f"Added secretary to list of created users")
            elif not secretary_user.is_validated:
                # If user exists but wasn't validated, validate them now
                secretary_user.is_validated = True
                secretary_user.validation_date = current_time
                secretary_user.save()
                print(f"Existing secretary user validated")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.secretary_email,
                    defaults={
                        'name': secretary_name,
                        'address': "Mettre a jour votre addresse",
                        'nationality': "Mettre a jour votre nationalité",
                        'birth_date': default_birth_date,
                        'job': "Association Secretary",
                        'joining_date': today,
                        'role': "Secrétaire générale",
                        'association': association
                    }
                )

                if member_created:
                    member.needs_profile_completion = True
                    member.save()
                    created_member_ids.append(member.id)
                    print(f"Created new Member entry for secretary: {secretary_name}")
                else:
                    # Update member with association secretary role if it already exists
                    if member.association != association or member.role != "Secrétaire générale":
                        member.association = association
                        member.role = "Secrétaire générale"
                        member.save()
                        print(f"Updated existing Member entry for secretary: {secretary_name}")
            except Exception as e:
                print(f"Error creating member for secretary: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"Total new users created: {len(created_users)}")
        print(f"Total new members created/flagged: {len(created_member_ids)}")

        # Send password reset emails to all newly created users
        for i, user in enumerate(created_users):
            print(f"Sending password reset email to user {i + 1}/{len(created_users)}: {user.email}")
            try:
                self._send_password_setup_email(user)
                print(f"Successfully initiated password reset for {user.email}")
            except Exception as e:
                print(f"Error sending password reset email to {user.email}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Return created member IDs for potential use in notifications
        return created_member_ids

    def _send_password_setup_email(self, user):
        """
        Send password reset email to newly created user
        """
        from django_rest_passwordreset.models import ResetPasswordToken
        from django.template.loader import render_to_string
        from django.core.mail import EmailMultiAlternatives
        from django.utils.html import strip_tags

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

        print(token_key)
        print(full_link)

        context = {
            'full_link': full_link,
            'email_adress': user.email,
            'full_name': user.full_name or user.email
        }

        html_message = render_to_string("backend/email.html", context=context)
        plain_message = strip_tags(html_message)

        msg = EmailMultiAlternatives(
            subject="Request for resetting password for {title}".format(title=user.email),
            body=plain_message,
            from_email="sender@example.com",
            to=[user.email]
        )

        msg.attach_alternative(html_message, "text/html")
        msg.send()
        print("Password reset email sent")

    def perform_update(self, serializer):
        """Override update to attempt automatic verification if relevant fields changed"""
        # Get the original instance
        instance = self.get_object()
        was_verified_before = instance.is_verified

        # Save the updates
        association = serializer.save()

        # Check if relevant fields changed
        rne_changed = instance.rne_document != association.rne_document
        matricule_changed = instance.matricule_fiscal != association.matricule_fiscal

        # If either field changed and both are present, verify again
        if (rne_changed or matricule_changed) and association.rne_document and association.matricule_fiscal:
            verified_association = self._verify_association(association)

            # If association just became verified, create user accounts
            if verified_association.is_verified and not was_verified_before:
                self._create_role_based_users(verified_association)

    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        """
        Endpoint to manually trigger verification for a specific association
        """
        association = self.get_object()
        was_verified_before = association.is_verified

        # Verify the association
        verified_association = self._verify_association(association)

        # If association just became verified, create user accounts
        if verified_association.is_verified and not was_verified_before:
            self._create_role_based_users(verified_association)

        # Return the updated verification status
        serializer = AssociationVerificationSerializer(verified_association)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def manual_verify(self, request, pk=None):
        """
        Endpoint for admins to manually set verification status
        """
        association = self.get_object()
        was_verified_before = association.is_verified

        # Update verification fields from request data
        verification_status = request.data.get('verification_status')
        verification_notes = request.data.get('verification_notes', '')

        if verification_status in ['pending', 'verified', 'failed']:
            association.verification_status = verification_status
            association.verification_notes = verification_notes

            # Update is_verified based on status
            association.is_verified = (verification_status == 'verified')

            # Set verification date if verified
            if verification_status == 'verified':
                association.verification_date = timezone.now()

            association.save()

            # If association just became verified, create user accounts
            if association.is_verified and not was_verified_before:
                self._create_role_based_users(association)

            serializer = AssociationVerificationSerializer(association)
            return Response(serializer.data)
        else:
            return Response(
                {"error": "Invalid verification status"},
                status=status.HTTP_400_BAD_REQUEST
            )


class UserProfileViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = UserProfileSerializer  # Make sure this is defined

    def list(self, request):
        """
        Get the current user's profile information
        """
        user = request.user
        serializer = UserProfileSerializer(user)
        return Response(serializer.data)

    def update(self, request, pk=None):
        """
        Update user profile information
        """
        # Don't try to use get_queryset() here since it's not defined
        # Instead, use the user from the request
        user = request.user

        print(f"Update request for user {user.id} with data: {request.data}")

        serializer = UserProfileSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            print(f"Serializer is valid. Validated data: {serializer.validated_data}")
            updated_user = serializer.save()
            print(f"User updated: {updated_user.id}")
            return Response(serializer.data)
        else:
            print(f"Serializer validation failed: {serializer.errors}")
            return Response(serializer.errors, status=400)

# Logout View
class LogoutViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request):
        """ Logs out the user by deleting their authentication token. """
        try:
            AuthToken.objects.filter(user=request.user).delete()
            return Response({"detail": "Logged out successfully"}, status=200)
        except Exception as e:
            return Response({"detail": str(e)}, status=400)


# Login View
class LoginViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = LoginSerializer

    def create(self, request):
        print(f"Login request: {request.data}")
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            user = authenticate(request, email=email, password=password)
            if user:
                is_admin_role = user.is_superuser or (
                        user.role and user.role.name in ['president', 'treasurer', 'secretary'])
                if user.is_validated or is_admin_role:
                    # Login successful
                    _, token = AuthToken.objects.create(user)
                    response_data = {
                        "user": self.serializer_class(user).data,
                        "token": token
                    }
                    print(f"Login successful: {response_data}")
                    return Response(response_data)
                else:
                    # Pending validation
                    error_response = {
                        "error": "Your account is pending validation. Please wait for approval from an administrator."}
                    print(f"Login rejected - pending validation: {error_response}")
                    return Response(error_response, status=403)
            else:
                # Invalid credentials
                error_response = {"error": "Invalid credentials"}
                print(f"Login rejected - invalid credentials: {error_response}")
                return Response(error_response, status=401)
        # Invalid data
        print(f"Login rejected - invalid data: {serializer.errors}")
        return Response(serializer.errors, status=400)

class AssociationRegisterViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = AssociationRegisterSerializer

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            association = serializer.save()
            print(f"Association created: {association.name} (ID: {association.id})")

            # Perform verification if the required fields are present
            if association.rne_document and association.matricule_fiscal:
                print(f"Starting verification for association: {association.name}")
                from .document_extractor_utils import process_association_verification
                association = process_association_verification(association)
                print(
                    f"Verification completed. Status: {association.verification_status}, Is verified: {association.is_verified}")

                # Only create user accounts if verification was successful
                if association.is_verified:
                    print(f"Association verified. Creating user accounts for: {association.name}")
                    # Create instance of AssociationAccountViewSet and use its method
                    account_viewset = AssociationAccountViewSet()
                    account_viewset._create_role_based_users(association)
                else:
                    print(f"Association not verified. User accounts not created.")

            # Return the updated serializer data
            updated_serializer = self.serializer_class(association)
            return Response(updated_serializer.data, status=201)
        return Response(serializer.errors, status=400)


class RegisterViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer

    def create(self, request):
        data = request.data
        association_id = data.get("association_id")

        if not association_id:
            return Response({"error": "Association ID is required"}, status=400)

        # Fetch the association instance
        association = get_object_or_404(AssociationAccount, id=association_id)

        # Validate CIN format
        cin = data.get('cin')
        if not cin or not re.match(r'^\d{8}$', cin):
            return Response({"error": "Le CIN doit contenir exactement 8 chiffres"}, status=400)

        # Validate birth_date format and value
        birth_date = data.get('birth_date')
        if not birth_date:
            return Response({"error": "La date de naissance est requise"}, status=400)

        try:
            # Check if birth_date is a valid date
            birth_date_obj = datetime.strptime(birth_date, '%Y-%m-%d').date()

            # Check if user is at least 18 years old
            today = date.today()
            age = today.year - birth_date_obj.year - (
                    (today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))
            if age < 18:
                return Response({"error": "Vous devez avoir au moins 18 ans pour vous inscrire"}, status=400)
        except ValueError:
            return Response({"error": "Format de date invalide. Utilisez le format YYYY-MM-DD"}, status=400)

        # Create serializer with full data
        serializer = self.serializer_class(data=data)
        if serializer.is_valid():
            # Include full_name and cin in the saved user data (but not birth_date as it's not a field in the User model)
            user = serializer.save(
                association=association,
                full_name=data.get("full_name", ""),
                cin=cin
            )

            # Store birth_date in the user's session or as a custom attribute
            # This can be retrieved later when validating the user
            print(f"User registered with CIN: {cin} and birth_date: {birth_date}")

            # Note: We do NOT create a Member record here anymore
            # The Member record will be created only after validation

            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)

    def _create_role_based_users(self, association):
        """
        Create user accounts for president, treasurer, and secretary
        with appropriate roles using their full names, and create corresponding Member objects.
        These accounts are automatically verified.
        """
        User = get_user_model()
        print(f"Starting user creation for association: {association.name}")

        # Get the Role objects (creating them if they don't exist)
        president_role, _ = Role.objects.get_or_create(name='president')
        treasurer_role, _ = Role.objects.get_or_create(name='treasurer')
        secretary_role, _ = Role.objects.get_or_create(name='secretary')

        # Generate temporary random password
        import secrets
        from datetime import date
        import datetime
        temp_password = secrets.token_urlsafe(12)
        print(f"Generated temporary password for new users")

        # Create users with appropriate roles if emails provided
        created_users = []

        # Import Member model from members app
        from api.models import Member

        today = date.today()
        default_birth_date = datetime.date(2000, 1, 1)

        # List to store created member IDs for notification
        created_member_ids = []

        # Current time for validation timestamp
        current_time = timezone.now()

        if association.president_email:
            # Use provided name or generate default name
            president_name = association.president_name or f"President of {association.name}"
            print(f"Creating president user: {president_name} ({association.president_email})")

            # Create CustomUser first
            president_user, created = User.objects.get_or_create(
                email=association.president_email,
                defaults={
                    'association': association,
                    'full_name': president_name,
                    'role': president_role,
                    'is_validated': True,  # Set to True by default
                    'validation_date': current_time  # Set validation date
                }
            )

            # Only set password if this is a new user
            if created:
                print(f"New president user created. Setting password.")
                president_user.set_password(temp_password)
                president_user.save()
                created_users.append(president_user)
                print(f"Added president to list of created users")
            elif not president_user.is_validated:
                # If user exists but wasn't validated, validate them now
                president_user.is_validated = True
                president_user.validation_date = current_time
                president_user.save()
                print(f"Existing president user validated")

            # Create or update corresponding Member object
            try:
                # Instead of get_or_create, check if member exists first, to handle the cin validation differently
                try:
                    member = Member.objects.get(email=association.president_email)
                    member_created = False
                    print(f"Found existing member with email {association.president_email}")
                except Member.DoesNotExist:
                    # Create the member manually with explicit None for cin
                    member = Member(
                        email=association.president_email,
                        name=president_name,
                        address="Mettre a jour votre addresse",
                        nationality="Mettre a jour votre nationalité",
                        birth_date=default_birth_date,
                        job="Association President",
                        joining_date=today,
                        role="Président",
                        association=association,
                        cin=None,  # Explicitly set CIN to None
                        needs_profile_completion=True
                    )
                    # Save without validation to bypass the cin validator for auto-created members
                    member.save(force_insert=True)
                    member_created = True
                    print(f"Created new Member entry for president: {president_name}")

                if not member_created:
                    # Update member with association president role if it already exists
                    if member.association != association or member.role != "Président":
                        member.association = association
                        member.role = "Président"
                        member.needs_profile_completion = True
                        # Save without validation
                        member.save(force_update=True)
                        print(f"Updated existing Member entry for president: {president_name}")

                created_member_ids.append(member.id)
            except Exception as e:
                print(f"Error creating member for president: {str(e)}")
                import traceback
                traceback.print_exc()

        # Similar updates for treasurer with automatic verification
        if association.treasurer_email:
            # Use provided name or generate default name
            treasurer_name = association.treasurer_name or f"Treasurer of {association.name}"
            print(f"Creating treasurer user: {treasurer_name} ({association.treasurer_email})")

            # Create CustomUser first
            treasurer_user, created = User.objects.get_or_create(
                email=association.treasurer_email,
                defaults={
                    'association': association,
                    'full_name': treasurer_name,
                    'role': treasurer_role,
                    'is_validated': True,  # Set to True by default
                    'validation_date': current_time  # Set validation date
                }
            )

            if created:
                print(f"New treasurer user created. Setting password.")
                treasurer_user.set_password(temp_password)
                treasurer_user.save()
                created_users.append(treasurer_user)
                print(f"Added treasurer to list of created users")
            elif not treasurer_user.is_validated:
                # If user exists but wasn't validated, validate them now
                treasurer_user.is_validated = True
                treasurer_user.validation_date = current_time
                treasurer_user.save()
                print(f"Existing treasurer user validated")

            # Create or update corresponding Member object
            try:
                # Instead of get_or_create, check if member exists first, to handle the cin validation differently
                try:
                    member = Member.objects.get(email=association.treasurer_email)
                    member_created = False
                    print(f"Found existing member with email {association.treasurer_email}")
                except Member.DoesNotExist:
                    # Create the member manually with explicit None for cin
                    member = Member(
                        email=association.treasurer_email,
                        name=treasurer_name,
                        address="Mettre a jour votre addresse",
                        nationality="Mettre a jour votre nationalité",
                        birth_date=default_birth_date,
                        job="Association Treasurer",
                        joining_date=today,
                        role="Trésorier",
                        association=association,
                        cin=None,  # Explicitly set CIN to None
                        needs_profile_completion=True
                    )
                    # Save without validation to bypass the cin validator for auto-created members
                    member.save(force_insert=True)
                    member_created = True
                    print(f"Created new Member entry for treasurer: {treasurer_name}")

                if not member_created:
                    # Update member with association treasurer role if it already exists
                    if member.association != association or member.role != "Trésorier":
                        member.association = association
                        member.role = "Trésorier"
                        member.needs_profile_completion = True
                        # Save without validation
                        member.save(force_update=True)
                        print(f"Updated existing Member entry for treasurer: {treasurer_name}")

                created_member_ids.append(member.id)
            except Exception as e:
                print(f"Error creating member for treasurer: {str(e)}")
                import traceback
                traceback.print_exc()

        # Similar updates for secretary with automatic verification
        if association.secretary_email:
            # Use provided name or generate default name
            secretary_name = association.secretary_name or f"Secretary of {association.name}"
            print(f"Creating secretary user: {secretary_name} ({association.secretary_email})")

            # Create CustomUser first
            secretary_user, created = User.objects.get_or_create(
                email=association.secretary_email,
                defaults={
                    'association': association,
                    'full_name': secretary_name,
                    'role': secretary_role,
                    'is_validated': True,  # Set to True by default
                    'validation_date': current_time  # Set validation date
                }
            )

            if created:
                print(f"New secretary user created. Setting password.")
                secretary_user.set_password(temp_password)
                secretary_user.save()
                created_users.append(secretary_user)
                print(f"Added secretary to list of created users")
            elif not secretary_user.is_validated:
                # If user exists but wasn't validated, validate them now
                secretary_user.is_validated = True
                secretary_user.validation_date = current_time
                secretary_user.save()
                print(f"Existing secretary user validated")

            # Create or update corresponding Member object
            try:
                # Instead of get_or_create, check if member exists first, to handle the cin validation differently
                try:
                    member = Member.objects.get(email=association.secretary_email)
                    member_created = False
                    print(f"Found existing member with email {association.secretary_email}")
                except Member.DoesNotExist:
                    # Create the member manually with explicit None for cin
                    member = Member(
                        email=association.secretary_email,
                        name=secretary_name,
                        address="Mettre a jour votre addresse",
                        nationality="Mettre a jour votre nationalité",
                        birth_date=default_birth_date,
                        job="Association Secretary",
                        joining_date=today,
                        role="Secrétaire générale",
                        association=association,
                        cin=None,  # Explicitly set CIN to None
                        needs_profile_completion=True
                    )
                    # Save without validation to bypass the cin validator for auto-created members
                    member.save(force_insert=True)
                    member_created = True
                    print(f"Created new Member entry for secretary: {secretary_name}")

                if not member_created:
                    # Update member with association secretary role if it already exists
                    if member.association != association or member.role != "Secrétaire générale":
                        member.association = association
                        member.role = "Secrétaire générale"
                        member.needs_profile_completion = True
                        # Save without validation
                        member.save(force_update=True)
                        print(f"Updated existing Member entry for secretary: {secretary_name}")

                created_member_ids.append(member.id)
            except Exception as e:
                print(f"Error creating member for secretary: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"Total new users created: {len(created_users)}")
        print(f"Total new members created/flagged: {len(created_member_ids)}")

        # Send password reset emails to all newly created users
        for i, user in enumerate(created_users):
            print(f"Sending password reset email to user {i + 1}/{len(created_users)}: {user.email}")
            try:
                self._send_password_setup_email(user)
                print(f"Successfully initiated password reset for {user.email}")
            except Exception as e:
                print(f"Error sending password reset email to {user.email}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Return created member IDs for potential use in notifications
        return created_member_ids

# Fetch Users (Admins see their own association users)
from .models import Role
from .permissions import MembersPermission, has_permission
from rest_framework.decorators import permission_classes


# Add Role ViewSet
class RoleViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for listing available roles"""
    queryset = Role.objects.all()
    serializer_class = RoleSerializer
    permission_classes = [permissions.AllowAny]


# Update existing ViewSets with permission checks

class UserViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated, MembersPermission]
    serializer_class = UserProfileSerializer

    def list(self, request):
        User = get_user_model()
        if request.user.is_superuser:
            queryset = User.objects.all()
        else:
            # Ensure users only see members of their own association
            if request.user.association:
                queryset = User.objects.filter(association=request.user.association)
            else:
                # Users without an association (shouldn't happen, but as a safeguard)
                queryset = User.objects.none()

        # Filter by role if specified
        role = request.query_params.get('role')
        if role:
            queryset = queryset.filter(role__name=role)

        # Filter by validation status if specified
        validation_status = request.query_params.get('validation_status')
        if validation_status:
            is_validated = validation_status.lower() == 'validated'
            queryset = queryset.filter(is_validated=is_validated)

        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)

    # Add a new endpoint to validate users
    @action(detail=True, methods=['post'])
    def validate_user(self, request, pk=None):
        """Validate or reject a user and create corresponding Member record if validated"""
        # Enhanced debugging
        print(f"validate_user called for user ID: {pk}")
        print(f"Current user: {request.user.email}, is_superuser: {request.user.is_superuser}")
        print(f"Request data: {request.data}")
        if hasattr(request.user, 'role') and request.user.role:
            print(f"Current user role: {request.user.role.name}")
        else:
            print("Current user has no role assigned")

        # Get the user to validate
        User = get_user_model()
        try:
            user_to_validate = get_object_or_404(User, pk=pk)
            print(f"Found user to validate: {user_to_validate.email}, CIN: {user_to_validate.cin}")
        except Exception as e:
            print(f"Error finding user with ID {pk}: {str(e)}")
            return Response(
                {"error": f"User with ID {pk} not found"},
                status=404
            )

        # Check for same association (skip for superusers)
        if not request.user.is_superuser and hasattr(request.user, 'association') and request.user.association:
            if not hasattr(user_to_validate, 'association') or not user_to_validate.association:
                print(f"User to validate {user_to_validate.email} has no association")
                return Response(
                    {"error": "The user you are trying to validate is not associated with any organization"},
                    status=400
                )

            if user_to_validate.association.id != request.user.association.id:
                print(
                    f"Association mismatch: Current user: {request.user.association.id}, User to validate: {user_to_validate.association.id}")
                return Response(
                    {"error": "You don't have permission to validate users from other associations"},
                    status=403
                )

        # Get action type with validation
        action_type = request.data.get('action', 'validate')
        print(f"Action type: {action_type}")

        if action_type == 'validate':
            # Validate the user
            user_to_validate.is_validated = True
            user_to_validate.validated_by = request.user
            user_to_validate.validation_date = timezone.now()
            user_to_validate.save()
            print(f"User {user_to_validate.email} validated successfully")

            # Now create a corresponding Member entry
            try:
                # Import Member model
                from api.models import Member
                from datetime import date
                import datetime

                today = date.today()

                # Check if the user already has a CIN
                cin = user_to_validate.cin
                if not cin:
                    print(f"Warning: User {user_to_validate.email} doesn't have a CIN")

                # Check for existing member with this email
                try:
                    existing_member = Member.objects.get(email=user_to_validate.email)
                    print(f"Found existing member with email {user_to_validate.email} - skipping creation")

                    # If member exists but has different association, update it
                    if existing_member.association != user_to_validate.association:
                        existing_member.association = user_to_validate.association
                        existing_member.save()
                        print(f"Updated existing member's association for {user_to_validate.email}")

                except Member.DoesNotExist:
                    # No existing member, create a new one
                    # Set default birth date
                    birth_date = datetime.date(2000, 1, 1)  # Default birth date

                    # Check if any required field is missing
                    needs_profile_completion = True  # Always start as True

                    # Create new member with validated user data
                    new_member = Member.objects.create(
                        email=user_to_validate.email,
                        name=user_to_validate.full_name or user_to_validate.email,
                        cin=cin,
                        birth_date=birth_date,
                        address="Mettre a jour votre addresse",
                        nationality="Mettre a jour votre nationalité",
                        job="Mettre a jour votre Métier",
                        joining_date=today,
                        role="Membre",  # Default role for regular members
                        association=user_to_validate.association,
                        needs_profile_completion=needs_profile_completion  # Always true for new members
                    )
                    print(
                        f"Created new Member entry for validated user: {user_to_validate.email} with needs_profile_completion=True")

            except Exception as e:
                print(f"Error creating member record for validated user: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue even if member creation fails - user is still validated

            # Success response
            return Response({
                "message": f"User {user_to_validate.email} has been validated and added to members"
            })

        elif action_type == 'reject':
            user_to_validate.is_validated = False
            user_to_validate.save()
            print(f"User {user_to_validate.email} rejected")

            # Delete the user account if it's not in any special role
            # Important: Don't delete users with special roles
            if user_to_validate.role and user_to_validate.role.name in ['president', 'treasurer', 'secretary']:
                print(f"User {user_to_validate.email} has role {user_to_validate.role.name}, not deleting account")
            else:
                # First check for and delete any existing Member record
                try:
                    from api.models import Member
                    try:
                        existing_member = Member.objects.get(email=user_to_validate.email)
                        existing_member.delete()
                        print(f"Deleted existing member record for rejected user: {user_to_validate.email}")
                    except Member.DoesNotExist:
                        # No member record to delete
                        print(f"No member record found for rejected user: {user_to_validate.email}")
                except Exception as e:
                    print(f"Error checking/deleting member record: {str(e)}")

                # Now consider deleting the user account (only for regular members)
                try:
                    if not user_to_validate.role or user_to_validate.role.name == 'member':
                        user_to_validate.delete()
                        print(f"Deleted user account for rejected user: {user_to_validate.email}")
                        return Response({
                            "message": f"User {user_to_validate.email} has been rejected and removed from the system"
                        })
                except Exception as e:
                    print(f"Error deleting user account: {str(e)}")

            return Response({
                "message": f"User {user_to_validate.email} has been rejected"
            })
        else:
            return Response({"error": "Invalid action. Use 'validate' or 'reject'"}, status=400)


# Fetch List of Associations (For User Registration)
class AssociationListViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = AssociationAccountSerializer  # Changed to AssociationAccountSerializer to include verification fields

    def list(self, request):
        associations = AssociationAccount.objects.all()
        serializer = self.serializer_class(associations, many=True)
        return Response(serializer.data)