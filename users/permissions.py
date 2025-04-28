from django.http import Http404
from rest_framework.permissions import BasePermission


# Add a custom permission type for user validation
class PermissionType:
    VIEW = 'view'
    CREATE = 'create'
    EDIT = 'edit'
    DELETE = 'delete'
    FULL_ACCESS = 'full_access'
    VALIDATE_USER = 'validate_user'  # New permission type


# Updated permission mapping to include validate_user permission
ROLE_PERMISSIONS = {
    'president': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE,
                    PermissionType.VALIDATE_USER],  # Added validate_user
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
    },
    'treasurer': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW, PermissionType.VALIDATE_USER],  # Added validate_user
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE],
        'chatbot': [PermissionType.VIEW],
    },
    'secretary': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW, PermissionType.VALIDATE_USER],  # Added validate_user
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
        'reports': [],
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
        # Special handling for custom actions
        if hasattr(view, 'action') and view.action == 'validate_user':
            # For validate_user, only president, treasurer, secretary, and superusers can access
            if request.user.is_superuser:
                return True

            if hasattr(request.user, 'role') and request.user.role:
                return request.user.role.name in ['president', 'treasurer', 'secretary']

            return False

        # Default permission handling for standard methods
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
        # Special handling for custom actions that might require specific permissions
        if hasattr(view, 'action'):
            # Actions that require edit permission
            if view.action in ['add_attendees', 'add_agenda_items', 'mark_complete', 'create_recurring_instances']:
                return has_permission(request.user, 'meetings', PermissionType.EDIT)

            # Actions that might require admin/approver role
            if view.action in ['approve_report']:
                # Only president or secretary can approve reports
                if request.user.is_superuser:
                    return True
                if hasattr(request.user, 'role') and request.user.role:
                    return request.user.role.name in ['president', 'secretary']
                return False

        # Default permission handling based on HTTP method
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