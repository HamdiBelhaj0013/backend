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
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # Add association to the member automatically
            if request.user.association:
                serializer.save(association=request.user.association)
            else:
                serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)
        serializer = self.serializer_class(member)
        return Response(serializer.data)

    def update(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)
        serializer = self.serializer_class(member, data=request.data)
        if serializer.is_valid():
            # Preserve the association when updating
            if member.association:
                serializer.save(association=member.association)
            else:
                serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=400)

    def destroy(self, request, pk=None):
        queryset = self.get_queryset()
        member = get_object_or_404(queryset, pk=pk)
        member.delete()
        return Response(status=204)