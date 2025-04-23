class AssociationFilterMixin:
    """
    A mixin to filter querysets by the user's association

    This ensures users can only access data from their own association
    """

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