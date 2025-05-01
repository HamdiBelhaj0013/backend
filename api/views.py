from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from rest_framework import viewsets, permissions
from .serializers import *
from rest_framework.response import Response
from .models import *

# Import the permissions from users app
from users.permissions import ProjectsPermission, MembersPermission

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.conf import settings
import redis
import json


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def test_websocket_connection(request):
    """
    Debug endpoint to test WebSocket connection components
    """
    result = {
        "redis_connection": test_redis_connection(),
        "token_info": get_token_info(request),
        "channel_layers": test_channel_layers(),
    }

    return Response(result)


def test_redis_connection():
    """Test Redis connection"""
    try:
        # Get Redis config from settings
        redis_host = settings.CHANNEL_LAYERS['default']['CONFIG']['hosts'][0][0]
        redis_port = settings.CHANNEL_LAYERS['default']['CONFIG']['hosts'][0][1]

        # Try to connect to Redis
        r = redis.Redis(host=redis_host, port=redis_port, db=0, socket_timeout=2)

        # Test connection with PING
        if r.ping():
            # Get some basic info
            return {
                "status": "success",
                "message": "Successfully connected to Redis",
                "redis_info": {
                    "version": r.info().get('redis_version', 'unknown'),
                    "clients_connected": r.info().get('connected_clients', 0),
                    "used_memory": r.info().get('used_memory_human', 'unknown')
                }
            }
        else:
            return {
                "status": "error",
                "message": "Redis PING failed"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Redis connection failed: {str(e)}"
        }


def get_token_info(request):
    """Get information about the token being used"""
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')

    # Basic info first
    token_info = {
        "auth_header_present": bool(auth_header),
        "auth_header_format": None,
    }

    # If we have an auth header, analyze it
    if auth_header:
        if auth_header.startswith('Token '):
            token_info["auth_header_format"] = "Token"
            token = auth_header[6:]  # Remove 'Token ' prefix
        elif auth_header.startswith('Bearer '):
            token_info["auth_header_format"] = "Bearer"
            token = auth_header[7:]  # Remove 'Bearer ' prefix
        else:
            token_info["auth_header_format"] = "Unknown"
            token = auth_header

        # Add token details
        token_info["token_length"] = len(token)
        token_info["token_preview"] = token[:10] + "..." if len(token) > 10 else token

        # Try to find token in Knox tokens
        from knox.models import AuthToken
        knox_token_exists = AuthToken.objects.filter(token_key=token[:8]).exists()
        token_info["knox_token_found"] = knox_token_exists

        # Check if there's a user ID in localStorage
        user_id = request.user.id if request.user.is_authenticated else None
        token_info["user_authenticated"] = request.user.is_authenticated
        token_info["user_id"] = user_id

    return token_info


def test_channel_layers():
    """Test if channel layers are configured correctly"""
    try:
        from channels.layers import get_channel_layer

        channel_layer = get_channel_layer()

        return {
            "status": "success",
            "channel_layer_type": channel_layer.__class__.__name__,
            "capacity": settings.CHANNEL_LAYERS['default']['CONFIG'].get('capacity', 100),
            "config": settings.CHANNEL_LAYERS['default']['CONFIG']
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Channel layers test failed: {str(e)}"
        }
class ProjectViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]  # Simplified for now
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

    def get_queryset(self):
        """Get projects for the current user's association only"""
        if self.request.user.is_superuser:
            return Project.objects.all()

        if self.request.user.association:
            # Filter projects by association
            return Project.objects.filter(association=self.request.user.association)

        return Project.objects.none()

    def list(self, request):
        queryset = self.get_queryset()
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # Add association to the project automatically
            if request.user.association:
                serializer.save(association=request.user.association)
            else:
                serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        project = get_object_or_404(queryset, pk=pk)
        serializer = self.serializer_class(project)
        return Response(serializer.data)

    def update(self, request, pk=None):
        queryset = self.get_queryset()
        project = get_object_or_404(queryset, pk=pk)
        serializer = self.serializer_class(project, data=request.data)
        if serializer.is_valid():
            # Preserve the association when updating
            if project.association:
                serializer.save(association=project.association)
            else:
                serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def destroy(self, request, pk=None):
        queryset = self.get_queryset()
        project = get_object_or_404(queryset, pk=pk)
        project.delete()
        return Response(status=204)


class MemberViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]  # Simplified for now
    queryset = Member.objects.all()
    serializer_class = MemberSerializer

    def get_queryset(self):
        """Get members for the current user's association only"""
        if self.request.user.is_superuser:
            return Member.objects.all()

        if self.request.user.association:
            return Member.objects.filter(association=self.request.user.association)

        return Member.objects.none()

    def list(self, request):
        queryset = self.get_queryset()
        serializer = self.serializer_class(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        print("Create Member Data:", request.data)
        # Ensure association is included in the data
        data = request.data.copy()

        # If user has an association, add it to the request data
        if request.user.association:
            # Create serializer with context and explicitly provide association ID in data
            data['association'] = request.user.association.id
            serializer = self.serializer_class(data=data, context={'request': request})

            if serializer.is_valid():
                member = serializer.save(
                    needs_profile_completion=data.get('needs_profile_completion', False)
                )
                return Response(serializer.data)
            else:
                print(f"Create validation errors: {serializer.errors}")
                return Response(serializer.errors, status=400)
        else:
            # User doesn't have an association
            return Response({"error": "You must be associated with an organization to create members"}, status=403)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)
        serializer = self.serializer_class(member)
        return Response(serializer.data)

    def update(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)

        print("Update Member Data:", request.data)
        # Use partial=True to allow partial updates without requiring all fields
        data = request.data.copy()
        data['association'] = member.association.id  # Ensure association ID is in data

        serializer = self.serializer_class(member, data=data, partial=True, context={'request': request})

        if serializer.is_valid():
            # Save with needs_profile_completion from request or keep existing value
            serializer.save(
                needs_profile_completion=request.data.get('needs_profile_completion', member.needs_profile_completion)
            )
            return Response(serializer.data)
        else:
            print(f"Update validation errors: {serializer.errors}")
            return Response(serializer.errors, status=400)

    def destroy(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)
        member.delete()
        return Response(status=204)