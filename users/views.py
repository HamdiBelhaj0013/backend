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

from .models import AssociationAccount
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
            self._verify_association(association)

    def _verify_association(self, association):
        """
        Internal method to verify an association's documents
        """
        from .document_extractor_utils import process_association_verification
        return process_association_verification(association)

    def perform_update(self, serializer):
        """Override update to attempt automatic verification if relevant fields changed"""
        # Get the original instance
        instance = self.get_object()

        # Save the updates
        association = serializer.save()

        # Check if relevant fields changed
        rne_changed = instance.rne_document != association.rne_document
        matricule_changed = instance.matricule_fiscal != association.matricule_fiscal

        # If either field changed and both are present, verify again
        if (rne_changed or matricule_changed) and association.rne_document and association.matricule_fiscal:
            self._verify_association(association)

    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        """
        Endpoint to manually trigger verification for a specific association
        """
        association = self.get_object()

        # Verify the association
        self._verify_association(association)

        # Return the updated verification status
        serializer = AssociationVerificationSerializer(association)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def manual_verify(self, request, pk=None):
        """
        Endpoint for admins to manually set verification status
        """
        association = self.get_object()

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
class AssociationRegisterViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = AssociationRegisterSerializer

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            association = serializer.save()

            # Perform verification if the required fields are present
            if association.rne_document and association.matricule_fiscal:
                from .document_extractor_utils import process_association_verification
                association = process_association_verification(association)

            # Return the updated serializer data
            updated_serializer = self.serializer_class(association)
            return Response(updated_serializer.data, status=201)
        return Response(serializer.errors, status=400)


# User Registration (Must belong to an Association)
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
class UserViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserProfileSerializer  # Changed to UserProfileSerializer to include full_name

    def list(self, request):
        User = get_user_model()
        if request.user.is_superuser:
            queryset = User.objects.all()  # Superusers see all users
        else:
            queryset = User.objects.filter(
                association=request.user.association)  # Regular users see only their association's users

        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)


# Fetch List of Associations (For User Registration)
class AssociationListViewset(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]
    serializer_class = AssociationAccountSerializer  # Changed to AssociationAccountSerializer to include verification fields

    def list(self, request):
        associations = AssociationAccount.objects.all()
        serializer = self.serializer_class(associations, many=True)
        return Response(serializer.data)