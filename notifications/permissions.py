from rest_framework import permissions


class IsOwnerOrAdmin(permissions.BasePermission):
    """
    Permission to allow only owners or admins to access notifications
    """

    def has_permission(self, request, view):
        # Must be authenticated to access notifications
        return request.user and request.user.is_authenticated

    def has_object_permission(self, request, view, obj):
        # Superusers can access any notification
        if request.user.is_superuser:
            return True

        # Admin users (president, treasurer, admins) can access any notification
        if hasattr(request.user, 'role') and request.user.role and request.user.role.name in ['admin', 'treasurer',
                                                                                              'president']:
            return True

        # Users can only access notifications directed to them or their role
        user_role = request.user.role.name if hasattr(request.user, 'role') and request.user.role else None

        return (
                (obj.recipient is None) or
                (obj.recipient == request.user) or
                (obj.recipient_role is None) or
                (user_role and obj.recipient_role == user_role)
        )


class CanManageNotifications(permissions.BasePermission):
    """
    Permission to allow only admin users to manage notifications system
    """

    def has_permission(self, request, view):
        # Must be authenticated
        if not request.user or not request.user.is_authenticated:
            return False

        # Only superusers and admin roles can create/manage notifications
        if request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            return (
                    request.user.is_superuser or
                    (hasattr(request.user, 'role') and request.user.role and
                     request.user.role.name in ['admin', 'treasurer', 'president'])
            )

        # Anyone authenticated can retrieve notifications
        return True

    def has_object_permission(self, request, view, obj):
        # Superusers can do anything
        if request.user.is_superuser:
            return True

        # Admin users can manage all notifications
        if hasattr(request.user, 'role') and request.user.role and request.user.role.name in ['admin', 'treasurer',
                                                                                              'president']:
            return True

        # For letters, users can only manage ones they sent
        if hasattr(obj, 'sender'):
            return obj.sender == request.user

        return False