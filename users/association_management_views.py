"""
Views for Association Management by Presidents
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404

from .models import CustomUser, AssociationAccount, Role
from .serializers import (
    UserProfileSerializer, AssociationAccountSerializer,
    RoleSerializer, UserValidationSerializer
)
from .association_manager_service import AssociationManagerService


class AssociationManagementViewSet(viewsets.ViewSet):
    """
    ViewSet for Association Presidents to manage their association and users
    """
    permission_classes = [IsAuthenticated]

    def get_service(self):
        """Get the Association Manager Service for the current user"""
        try:
            return AssociationManagerService(self.request.user)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)

    def list(self, request):
        """
        List all users in the association with optional filtering
        """
        try:
            service = self.get_service()

            # Extract filters from query parameters
            filters = {}
            if 'role' in request.query_params:
                filters['role'] = request.query_params.get('role')
            if 'validated' in request.query_params:
                filters['is_validated'] = request.query_params.get('validated') == 'true'
            if 'email' in request.query_params:
                filters['email'] = request.query_params.get('email')
            if 'name' in request.query_params:
                filters['full_name'] = request.query_params.get('name')

            # Get filtered users
            users = service.get_users(filters)

            # Serialize and return
            serializer = UserProfileSerializer(users, many=True)
            return Response(serializer.data)

        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def association(self, request):
        """
        Get the association details
        """
        try:
            service = self.get_service()
            association = service.get_association_details()
            serializer = AssociationAccountSerializer(association)
            return Response(serializer.data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['put', 'patch'])
    def update_association(self, request):
        """
        Update the association details
        """
        try:
            service = self.get_service()
            association = service.update_association_details(request.data)
            serializer = AssociationAccountSerializer(association)
            return Response(serializer.data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def user(self, request, pk=None):
        """
        Get a specific user
        """
        try:
            service = self.get_service()
            user = service.get_user(pk)
            serializer = UserProfileSerializer(user)
            return Response(serializer.data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['put', 'patch'])
    def update_user(self, request, pk=None):
        """
        Update a specific user
        """
        try:
            service = self.get_service()
            user = service.update_user(pk, request.data)
            serializer = UserProfileSerializer(user)
            return Response(serializer.data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def validate_user(self, request, pk=None):
        """
        Validate or reject a user
        """
        try:
            service = self.get_service()
            validate = request.data.get('action', 'validate') == 'validate'
            user = service.validate_user(pk, validate)

            if user:
                serializer = UserProfileSerializer(user)
                return Response(serializer.data)
            else:
                # User was rejected and deleted
                return Response({
                    "message": "User has been rejected and removed from the system"
                })

        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def create_user(self, request):
        """
        Create a new user in the association
        """
        try:
            service = self.get_service()
            user, password = service.create_user(request.data)

            # Return user data and temporary password if one was generated
            serializer = UserProfileSerializer(user)
            response_data = serializer.data

            # Only include password in response if it was auto-generated
            if not request.data.get('password'):
                response_data['temporary_password'] = password

            return Response(response_data, status=status.HTTP_201_CREATED)

        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['delete'])
    def delete_user(self, request, pk=None):
        """
        Delete a user from the association
        """
        try:
            service = self.get_service()
            service.delete_user(pk)
            return Response({"message": "User deleted successfully"})
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def set_role(self, request, pk=None):
        """
        Set a user's role
        """
        try:
            service = self.get_service()
            role_id = request.data.get('role_id')

            if not role_id:
                return Response({"error": "role_id is required"}, status=status.HTTP_400_BAD_REQUEST)

            user = service.set_user_role(pk, role_id)
            serializer = UserProfileSerializer(user)
            return Response(serializer.data)

        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'])
    def reset_password(self, request, pk=None):
        """
        Send a password reset email to a user
        """
        try:
            service = self.get_service()
            service.send_password_reset(pk)
            return Response({"message": "Password reset email sent"})
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except CustomUser.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def roles(self, request):
        """
        Get all available roles
        """
        try:
            service = self.get_service()
            roles = service.get_roles()
            serializer = RoleSerializer(roles, many=True)
            return Response(serializer.data)
        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def members(self, request):
        """
        Get all members in the association
        """
        try:
            service = self.get_service()

            # Extract filters from query parameters
            filters = {}
            if 'role' in request.query_params:
                filters['role'] = request.query_params.get('role')
            if 'name' in request.query_params:
                filters['name'] = request.query_params.get('name')
            if 'email' in request.query_params:
                filters['email'] = request.query_params.get('email')
            if 'needs_profile_completion' in request.query_params:
                filters['needs_profile_completion'] = request.query_params.get('needs_profile_completion') == 'true'

            members = service.get_members(filters)

            # Import the member serializer from api
            from api.serializers import MemberSerializer
            serializer = MemberSerializer(members, many=True)
            return Response(serializer.data)

        except PermissionDenied as e:
            return Response({"error": str(e)}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)