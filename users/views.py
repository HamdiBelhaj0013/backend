from django.shortcuts import get_object_or_404
from rest_framework import viewsets, permissions
from django.contrib.auth import get_user_model, authenticate
from knox.models import AuthToken
from .serializers import *
from .serializers import UserProfileSerializer
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.utils import timezone

from .models import AssociationAccount, Role, CustomUser
from .serializers import AssociationAccountSerializer, AssociationVerificationSerializer
from .document_extractor_utils import verify_association_document


class AssociationAccountViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing association accounts
    """
    queryset = AssociationAccount.objects.all()
    serializer_class = AssociationAccountSerializer

    def get_permissions(self):
        """
        Custom permissions:
        - Admin users can access all endpoints
        - Regular users can only create and view their own account
        - Allow anonymous access for retrieve (GET) to check verification status
        """
        # Allow anonymous access for retrieve (GET) operation
        if self.action == 'retrieve':
            return [permissions.AllowAny()]

        # Admin-only actions
        if self.action in ['list', 'verify', 'manual_verify']:
            return [IsAdminUser()]

        return [IsAuthenticated()]

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

    def _create_role_based_users(self, association):
        """
        Create user accounts for president, treasurer, and secretary
        with appropriate roles
        """
        User = get_user_model()

        # Get the Role objects (creating them if they don't exist)
        president_role, _ = Role.objects.get_or_create(name='president')
        treasurer_role, _ = Role.objects.get_or_create(name='treasurer')
        secretary_role, _ = Role.objects.get_or_create(name='secretary')

        # Create temporary random passwords - these will be reset
        # by the users during their first login
        import secrets
        temp_password = secrets.token_urlsafe(12)

        # Create users with appropriate roles if emails provided
        if association.president_email:
            president_user, created = User.objects.get_or_create(
                email=association.president_email,
                defaults={
                    'password': temp_password,
                    'association': association,
                    'full_name': f"President of {association.name}",
                    'role': president_role
                }
            )
            # Only send email if this is a new user
            if created:
                # Hash password properly for new user
                president_user.set_password(temp_password)
                president_user.save()
                # Send password reset email to set password
                self._send_password_setup_email(president_user)

        if association.treasurer_email:
            treasurer_user, created = User.objects.get_or_create(
                email=association.treasurer_email,
                defaults={
                    'password': temp_password,
                    'association': association,
                    'full_name': f"Treasurer of {association.name}",
                    'role': treasurer_role
                }
            )
            if created:
                treasurer_user.set_password(temp_password)
                treasurer_user.save()
                # Send password reset email to set password
                self._send_password_setup_email(treasurer_user)

        if association.secretary_email:
            secretary_user, created = User.objects.get_or_create(
                email=association.secretary_email,
                defaults={
                    'password': temp_password,
                    'association': association,
                    'full_name': f"Secretary of {association.name}",
                    'role': secretary_role
                }
            )
            if created:
                secretary_user.set_password(temp_password)
                secretary_user.save()
                # Send password reset email to set password
                self._send_password_setup_email(secretary_user)

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
        user = request.user
        serializer = UserProfileSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
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
                _, token = AuthToken.objects.create(user)
                return Response(
                    {
                        "user": self.serializer_class(user).data,
                        "token": token
                    }
                )
            else:
                return Response({"error": "Invalid credentials"}, status=401)
        return Response(serializer.errors, status=400)


# Association Registration View
# Association Registration View
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
                    self._create_role_based_users(association)
                else:
                    print(f"Association not verified. User accounts not created.")

            # Return the updated serializer data
            updated_serializer = self.serializer_class(association)
            return Response(updated_serializer.data, status=201)
        return Response(serializer.errors, status=400)

    def _create_role_based_users(self, association):
        """
        Create user accounts for president, treasurer, and secretary
        with appropriate roles using their full names
        """
        User = get_user_model()
        print(f"Starting user creation for association: {association.name}")

        # Get the Role objects (creating them if they don't exist)
        president_role, _ = Role.objects.get_or_create(name='president')
        treasurer_role, _ = Role.objects.get_or_create(name='treasurer')
        secretary_role, _ = Role.objects.get_or_create(name='secretary')

        # Create temporary random passwords - these will be reset
        # by the users during their first login
        import secrets
        temp_password = secrets.token_urlsafe(12)
        print(f"Generated temporary password for new users")

        # Create users with appropriate roles if emails provided
        created_users = []

        if association.president_email:
            # Use provided name or generate default name
            president_name = association.president_name or f"President of {association.name}"
            print(f"Creating president user: {president_name} ({association.president_email})")
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
            else:
                print(f"President user already exists. Skipping.")

        if association.treasurer_email:
            # Use provided name or generate default name
            treasurer_name = association.treasurer_name or f"Treasurer of {association.name}"
            print(f"Creating treasurer user: {treasurer_name} ({association.treasurer_email})")
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
            else:
                print(f"Treasurer user already exists. Skipping.")

        if association.secretary_email:
            # Use provided name or generate default name
            secretary_name = association.secretary_name or f"Secretary of {association.name}"
            print(f"Creating secretary user: {secretary_name} ({association.secretary_email})")
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
            else:
                print(f"Secretary user already exists. Skipping.")

        print(f"Total new users created: {len(created_users)}")

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

    def _send_password_setup_email(self, user):
        """
        Send password reset email to newly created user
        """
        from django_rest_passwordreset.models import ResetPasswordToken
        from django.template.loader import render_to_string
        from django.core.mail import EmailMultiAlternatives
        from django.utils.html import strip_tags

        print(f"Creating password reset token for user: {user.email}")

        # Create password reset token
        try:
            token = ResetPasswordToken.objects.create(
                user=user,
                user_agent="API",
                ip_address="127.0.0.1"
            )
            print(f"Token created successfully: {token.key}")

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

            print(f"Rendering email template for {user.email}")
            html_message = render_to_string("backend/email.html", context=context)
            plain_message = strip_tags(html_message)

            print(f"Preparing email message for {user.email}")
            msg = EmailMultiAlternatives(
                subject="Request for resetting password for {title}".format(title=user.email),
                body=plain_message,
                from_email="sender@example.com",
                to=[user.email]
            )

            msg.attach_alternative(html_message, "text/html")
            print(f"Sending email to {user.email}")
            msg.send()
            print("Password reset email sent")

        except Exception as e:
            print(f"Error in _send_password_setup_email: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


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
            queryset = User.objects.filter(association=request.user.association)

        # Filter by role if specified
        role = request.query_params.get('role')
        if role:
            queryset = queryset.filter(role__name=role)

        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)

    # Add ability to update a user's role (for presidents)
    @action(detail=True, methods=['post'])
    def assign_role(self, request, pk=None):
        # Check if user has permission to manage roles
        if not has_permission(request.user, 'members', 'edit'):
            return Response({"error": "You don't have permission to assign roles"}, status=403)

        user = get_object_or_404(get_user_model(), pk=pk)
        role_id = request.data.get('role_id')

        if not role_id:
            return Response({"error": "Role ID is required"}, status=400)

        role = get_object_or_404(Role, pk=role_id)

        # Update the user's role
        user.role = role
        user.save()

        serializer = self.serializer_class(user)
        return Response(serializer.data)


# Fetch List of Associations (For User Registration)
class AssociationListViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = AssociationAccountSerializer  # Changed to AssociationAccountSerializer to include verification fields

    def list(self, request):
        associations = AssociationAccount.objects.all()
        serializer = self.serializer_class(associations, many=True)
        return Response(serializer.data)