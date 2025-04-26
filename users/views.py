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
        with appropriate roles using their full names, and create corresponding Member objects
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
                    'role': president_role
                }
            )

            # Only set password if this is a new user
            if created:
                print(f"New president user created. Setting password.")
                president_user.set_password(temp_password)
                president_user.save()
                created_users.append(president_user)
                print(f"Added president to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.president_email,
                    defaults={
                        'name': president_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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

        # Similar updates for treasurer and secretary with the needs_profile_completion flag
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
                    'role': treasurer_role
                }
            )

            if created:
                print(f"New treasurer user created. Setting password.")
                treasurer_user.set_password(temp_password)
                treasurer_user.save()
                created_users.append(treasurer_user)
                print(f"Added treasurer to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.treasurer_email,
                    defaults={
                        'name': treasurer_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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
                    'role': secretary_role
                }
            )

            if created:
                print(f"New secretary user created. Setting password.")
                secretary_user.set_password(temp_password)
                secretary_user.save()
                created_users.append(secretary_user)
                print(f"Added secretary to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.secretary_email,
                    defaults={
                        'name': secretary_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            user = authenticate(request, email=email, password=password)
            if user:
                # Check if the user is validated or has a role that doesn't need validation
                is_admin_role = user.is_superuser or (
                            user.role and user.role.name in ['president', 'treasurer', 'secretary'])

                if user.is_validated or is_admin_role:
                    _, token = AuthToken.objects.create(user)
                    return Response(
                        {
                            "user": self.serializer_class(user).data,
                            "token": token
                        }
                    )
                else:
                    return Response({
                                        "error": "Your account is pending validation. Please wait for approval from an administrator."},
                                    status=403)
            else:
                return Response({"error": "Invalid credentials"}, status=401)
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

        serializer = self.serializer_class(data=data)
        if serializer.is_valid():
            # Include full_name in the saved user data if provided
            full_name = data.get("full_name", "")
            user = serializer.save(association=association, full_name=full_name)
            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)
    def _create_role_based_users(self, association):
        """
        Create user accounts for president, treasurer, and secretary
        with appropriate roles using their full names, and create corresponding Member objects
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
                    'role': president_role
                }
            )

            # Only set password if this is a new user
            if created:
                print(f"New president user created. Setting password.")
                president_user.set_password(temp_password)
                president_user.save()
                created_users.append(president_user)
                print(f"Added president to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.president_email,
                    defaults={
                        'name': president_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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

        # Similar updates for treasurer and secretary with the needs_profile_completion flag
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
                    'role': treasurer_role
                }
            )

            if created:
                print(f"New treasurer user created. Setting password.")
                treasurer_user.set_password(temp_password)
                treasurer_user.save()
                created_users.append(treasurer_user)
                print(f"Added treasurer to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.treasurer_email,
                    defaults={
                        'name': treasurer_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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
                    'role': secretary_role
                }
            )

            if created:
                print(f"New secretary user created. Setting password.")
                secretary_user.set_password(temp_password)
                secretary_user.save()
                created_users.append(secretary_user)
                print(f"Added secretary to list of created users")

            # Create or update corresponding Member object
            try:
                member, member_created = Member.objects.get_or_create(
                    email=association.secretary_email,
                    defaults={
                        'name': secretary_name,
                        'address': "Please update your address",
                        'nationality': "Please update your nationality",
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
        """Validate or reject a user"""
        # Check if user has permission to validate members
        if not (request.user.is_superuser or
                (request.user.role and request.user.role.name in ['president', 'treasurer', 'secretary'])):
            return Response(
                {"error": "You don't have permission to validate users"},
                status=403
            )

        User = get_user_model()
        user_to_validate = get_object_or_404(User, pk=pk)

        # Make sure the user being validated belongs to the same association
        if user_to_validate.association != request.user.association and not request.user.is_superuser:
            return Response(
                {"error": "You don't have permission to validate users from other associations"},
                status=403
            )

        # Validate the user
        action_type = request.data.get('action', 'validate')

        if action_type == 'validate':
            user_to_validate.is_validated = True
            user_to_validate.validated_by = request.user
            user_to_validate.validation_date = timezone.now()
            user_to_validate.save()
            return Response({"message": f"User {user_to_validate.email} has been validated"})
        elif action_type == 'reject':
            user_to_validate.is_validated = False
            user_to_validate.save()
            return Response({"message": f"User {user_to_validate.email} has been rejected"})
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