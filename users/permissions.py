from django.http import Http404
from rest_framework.permissions import BasePermission


class PermissionType:
    VIEW = 'view'
    CREATE = 'create'
    EDIT = 'edit'
    DELETE = 'delete'
    FULL_ACCESS = 'full_access'


# Permission mapping by role and resource
ROLE_PERMISSIONS = {
    'president': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
    },
    'treasurer': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW],
        'meetings': [PermissionType.VIEW],
        'reports': [PermissionType.VIEW, PermissionType.CREATE],  # Can only create/view financial reports
        'chatbot': [PermissionType.VIEW],
    },
    'secretary': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW],
        'finance': [PermissionType.VIEW],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
    },
    'member': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW],
        'finance': [PermissionType.VIEW],
        'tasks': [PermissionType.VIEW],
        'meetings': [PermissionType.VIEW],
        'reports': [],  # No access to reports
        'chatbot': [PermissionType.VIEW],
    }
}


def has_permission(user, resource_type, permission_type):
    """
    Check if a user has a specific permission for a resource

    Args:
        user: Django user object
        resource_type: Type of resource (projects, members, etc.)
        permission_type: Type of permission (view, create, edit, delete)

    Returns:
        Boolean indicating if user has permission
    """
    # Superusers have all permissions
    if user.is_superuser:
        return True

    # If user has no role, deny permission
    if not hasattr(user, 'role') or user.role is None:
        return False

    role_name = user.role.name

    # Get permissions for this role and resource
    resource_permissions = ROLE_PERMISSIONS.get(role_name, {}).get(resource_type, [])

    # Check if the permission is in the list or if FULL_ACCESS is granted
    return permission_type in resource_permissions or PermissionType.FULL_ACCESS in resource_permissions


class MembersPermission(BasePermission):
    """Permission class specifically for checking member resource permissions"""

    def has_permission(self, request, view):
        # Map HTTP methods to permission types
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'members', permission_type)


class ProjectsPermission(BasePermission):
    """Permission class specifically for checking project resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'projects', permission_type)


class FinancePermission(BasePermission):
    """Permission class specifically for checking finance resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'finance', permission_type)


class TasksPermission(BasePermission):
    """Permission class specifically for checking tasks resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'tasks', permission_type)


class MeetingsPermission(BasePermission):
    """Permission class specifically for checking meetings resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'meetings', permission_type)


class ReportsPermission(BasePermission):
    """Permission class specifically for checking reports resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'reports', permission_type)


class ChatbotPermission(BasePermission):
    """Permission class specifically for checking chatbot resource permissions"""

    def has_permission(self, request, view):
        method_mapping = {
            'GET': PermissionType.VIEW,
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        return has_permission(request.user, 'chatbot', permission_type)