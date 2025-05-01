from django.db import models
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from django.db.models import Q

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


# Validator for CIN - should be exactly 8 digits
cin_validator = RegexValidator(
    regex=r'^\d{8}$',
    message='CIN doit contenir exactement 8 chiffres',
    code='invalid_cin'
)


class Member(models.Model):
    name = models.CharField(max_length=100)
    # Add CIN field with validation for 8 digits and uniqueness
    # Make it nullable for auto-created members, but with blank=False to require it in forms
    cin = models.CharField(
        max_length=8,
        validators=[cin_validator],
        unique=True,
        verbose_name="CIN",
        help_text="Carte d'Identité Nationale (8 chiffres)",
        null=True,
        blank=True  # Allow blank to fix validation errors for auto-created members
    )
    address = models.CharField(max_length=100)
    email = models.EmailField(default='example@example.com')  # Default value for email
    nationality = models.CharField(unique=False, max_length=100)
    birth_date = models.DateField()
    job = models.CharField(max_length=100)
    joining_date = models.DateField()
    # Role will be handled with custom validation in clean()
    role = models.CharField(max_length=100)
    association = models.ForeignKey(
        AssociationAccount,
        on_delete=models.CASCADE,
        related_name='members',
        null=True
    )
    # New field to indicate if member profile needs completion
    needs_profile_completion = models.BooleanField(default=False,
                                                   help_text="Indicates if this member needs to complete their profile")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def clean(self):
        super().clean()

        # Modified CIN validation - only validate when CIN is not None
        if self.cin is not None and not cin_validator.regex.match(self.cin):
            raise ValidationError({
                'cin': cin_validator.message
            })

        # Role validation only if association exists
        if self.role not in ['Membre', 'autre', 'Président', 'Trésorier', 'Secrétaire générale'] and self.association:
            existing = Member.objects.filter(
                ~Q(id=self.id),
                association=self.association,
                role=self.role
            ).exists()

            if existing:
                raise ValidationError({
                    'role': f"Le rôle '{self.role}' est déjà attribué à un autre membre dans cette association. "
                            f"Seuls les rôles 'Membre' et 'autre' peuvent être partagés."
                })

    def save(self, *args, **kwargs):
        # Skip validation if force_insert or force_update is set
        skip_validation = kwargs.pop('skip_validation', False)
        force_insert = kwargs.get('force_insert', False)
        force_update = kwargs.get('force_update', False)

        if skip_validation or force_insert or force_update:
            # Skip validation for system-created members
            super(Member, self).save(*args, **kwargs)
        else:
            # Regular validation for user-created members
            self.full_clean()
            super(Member, self).save(*args, **kwargs)

    class Meta:
        unique_together = ('name', 'association')