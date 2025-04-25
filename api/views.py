from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from rest_framework import viewsets, permissions
from .serializers import *
from rest_framework.response import Response
from .models import *

# Import the permissions from users app
from users.permissions import ProjectsPermission, MembersPermission


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
            # Don't modify the data dictionary directly
            serializer = self.serializer_class(data=data)
            if serializer.is_valid():
                serializer.save(
                    association=request.user.association,
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
        serializer = self.serializer_class(member, data=request.data, partial=True)

        if serializer.is_valid():
            # Always preserve the existing association
            serializer.save(
                association=member.association,
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