from django.shortcuts import get_object_or_404
from rest_framework import viewsets, permissions
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate
from knox.models import AuthToken
from .serializers import *
from .models import AssociationAccount

User = get_user_model()


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
            return Response(serializer.data, status=201)
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
            user = serializer.save(association=association)
            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)


# Fetch Users (Admins see their own association users)
class UserViewset(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = RegisterSerializer

    def list(self, request):
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
    serializer_class = AssociationRegisterSerializer  # Serializer for associations

    def list(self, request):
        associations = AssociationAccount.objects.all()
        serializer = self.serializer_class(associations, many=True)
        return Response(serializer.data)
