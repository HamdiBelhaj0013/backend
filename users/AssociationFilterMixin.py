from rest_framework import viewsets
class AssociationFilterMixin:


    def get_queryset(self):
        """Filter queryset based on user's association"""
        queryset = super().get_queryset()

        # Superusers can see all records
        if self.request.user.is_superuser:
            return queryset

        # Non-superusers can only see records from their association
        if self.request.user.association:
            return queryset.filter(association=self.request.user.association)

        # Users without an association shouldn't see any records
        return queryset.none()


# Example usage for a ViewSet:
class SecureUserViewSet(AssociationFilterMixin, viewsets.ModelViewSet):
    # Your existing viewset code
    pass


class AssociationFilterMixin:
    """
    A mixin that filters querysets by the user's association.
    Ensures data isolation between different associations.
    """

    def get_queryset(self):
        """Filter queryset to only include records from the user's association"""
        queryset = super().get_queryset()

        # Superusers can see all records
        if self.request.user.is_superuser:
            return queryset

        # Regular users can only see records from their association
        if hasattr(self.request.user, 'association') and self.request.user.association:
            # For models with direct association field
            if hasattr(queryset.model, 'association'):
                return queryset.filter(association=self.request.user.association)

            # For models with project that has association
            elif hasattr(queryset.model, 'project'):
                return queryset.filter(project__association=self.request.user.association)

            # For models with generated_by (like reports) that's linked to a user with association
            elif hasattr(queryset.model, 'generated_by'):
                return queryset.filter(generated_by__association=self.request.user.association)

            # For Donor model that might have transactions with projects from user's association
            elif queryset.model.__name__ == 'Donor':
                from api.models import Project
                association_projects = Project.objects.filter(
                    association=self.request.user.association
                ).values_list('id', flat=True)

                return queryset.filter(
                    transactions__project__in=association_projects
                ).distinct()

        # Users without association can't see any records
        return queryset.none()

    def perform_create(self, serializer):
        """Add association to new records automatically"""
        if hasattr(self.request.user, 'association') and self.request.user.association:
            serializer.save(created_by=self.request.user, association=self.request.user.association)
        else:
            serializer.save(created_by=self.request.user)