from django.http import Http404
from rest_framework.permissions import BasePermission


class PermissionType:
    VIEW = 'view'
    CREATE = 'create'
    EDIT = 'edit'
    DELETE = 'delete'
    FULL_ACCESS = 'full_access'
    VALIDATE_USER = 'validate_user'


ROLE_PERMISSIONS = {
    'president': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE,
                    PermissionType.VALIDATE_USER],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
        'notifications': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
    },
    'treasurer': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE,
                   PermissionType.VALIDATE_USER],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
        'notifications': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
    },
    'secretary': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE,
                   PermissionType.VALIDATE_USER],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
        'notifications': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
    },
    'member': {
        'projects': [PermissionType.VIEW],
        'members': [PermissionType.VIEW],
        'finance': [PermissionType.VIEW],
        'tasks': [PermissionType.VIEW],
        'meetings': [PermissionType.VIEW],
        'reports': [],
        'chatbot': [PermissionType.VIEW],
        'notifications': [PermissionType.VIEW],
    },
    # Map admin role (from backend) to match president permissions
    'admin': {
        'projects': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'members': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE,
                    PermissionType.VALIDATE_USER],
        'finance': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'tasks': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'meetings': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'reports': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
        'chatbot': [PermissionType.VIEW],
        'notifications': [PermissionType.VIEW, PermissionType.CREATE, PermissionType.EDIT, PermissionType.DELETE],
    }
}

def has_permission(user, resource_type, permission_type):


    if user.is_superuser:
        return True


    if not hasattr(user, 'role') or user.role is None:
        return False

    role_name = user.role.name


    resource_permissions = ROLE_PERMISSIONS.get(role_name, {}).get(resource_type, [])

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


class NotificationPermission(BasePermission):
    """Permission class specifically for checking notification resource permissions"""

    def has_permission(self, request, view):
        # All authenticated users can view notifications
        if request.method == 'GET':
            return request.user and request.user.is_authenticated

        # For other operations, check specific permissions
        method_mapping = {
            'POST': PermissionType.CREATE,
            'PUT': PermissionType.EDIT,
            'PATCH': PermissionType.EDIT,
            'DELETE': PermissionType.DELETE,
        }

        permission_type = method_mapping.get(request.method)
        if permission_type is None:
            return False

        # Handle role mapping between 'admin' and 'president'
        if hasattr(request.user, 'role'):
            # Map 'admin' backend role to 'president' frontend role for permission checking
            role_name = request.user.role.name
            if role_name == 'admin':
                return has_permission(request.user, 'notifications', permission_type) or has_permission(request.user,
                                                                                                        'admin',
                                                                                                        permission_type)

        return has_permission(request.user, 'notifications', permission_type)