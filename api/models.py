from django.db import models

# Import the AssociationAccount model from users app
from users.models import AssociationAccount


class Project(models.Model):
    name = models.CharField(unique=True, max_length=100)
    start_date = models.DateField()
    end_date = models.DateField()
    description = models.TextField(max_length=500)
    budget = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=100)
    # Update association field related_name
    association = models.ForeignKey(
        AssociationAccount,
        on_delete=models.CASCADE,
        related_name='projects',  # CHANGED FROM 'members' TO 'projects'
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Member(models.Model):
    name = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    email = models.EmailField(default='example@example.com')  # Default value for email
    nationality = models.CharField(unique=False, max_length=100)
    birth_date = models.DateField()
    job = models.CharField(unique=False, max_length=100)
    joining_date = models.DateField()
    role = models.CharField(unique=False, max_length=100)
    # Add association field to link members to associations
    association = models.ForeignKey(
        AssociationAccount,
        on_delete=models.CASCADE,
        related_name='members',
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        # Update uniqueness constraint to be per-association
        unique_together = ('name', 'association')