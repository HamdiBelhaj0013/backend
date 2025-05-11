from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.dispatch import receiver
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags
from django_rest_passwordreset.signals import reset_password_token_created
from django.core.validators import RegexValidator

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, association=None, full_name=None, **extra_fields):
        if not email:
            raise ValueError('Email is a required field')


        if not association and not extra_fields.get('is_superuser', False):
            raise ValueError('Association is a required field for regular users')

        email = self.normalize_email(email)
        extra_fields.setdefault('is_active', True)

        user = self.model(email=email, association=association, full_name=full_name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, association=None, full_name=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, association, full_name, **extra_fields)


class AssociationAccount(models.Model):
    name = models.CharField(max_length=255, unique=True)
    email = models.EmailField(unique=True)

    # President information
    president_email = models.EmailField(blank=True, null=True, help_text="Email of the association's president")
    president_name = models.CharField(max_length=255, blank=True, null=True,
                                      help_text="Full name of the association's president")

    # Treasurer information
    treasurer_email = models.EmailField(blank=True, null=True, help_text="Email of the association's treasurer")
    treasurer_name = models.CharField(max_length=255, blank=True, null=True,
                                      help_text="Full name of the association's treasurer")

    # Secretary information
    secretary_email = models.EmailField(blank=True, null=True, help_text="Email of the association's general secretary")
    secretary_name = models.CharField(max_length=255, blank=True, null=True,
                                      help_text="Full name of the association's general secretary")

    cin_recto = models.FileField(upload_to='documents/cin/', blank=True, null=True)
    cin_verso = models.FileField(upload_to='documents/cin/', blank=True, null=True)
    matricule_fiscal = models.CharField(max_length=100, unique=True)
    rne_document = models.FileField(upload_to='documents/rne/', blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    verification_date = models.DateTimeField(null=True, blank=True)
    verification_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('verified', 'Verified'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    verification_notes = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name


class Role(models.Model):
    """Role model for defining user roles within an association"""
    ROLE_CHOICES = (
        ('president', 'President'),
        ('treasurer', 'Treasurer'),
        ('secretary', 'General Secretary'),
        ('member', 'Regular Member'),
    )

    name = models.CharField(max_length=20, choices=ROLE_CHOICES)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.get_name_display()


cin_validator = RegexValidator(
    regex=r'^\d{8}$',
    message='CIN doit contenir exactement 8 chiffres',
    code='invalid_cin'
)

class CustomUser(AbstractUser):
    email = models.EmailField(max_length=200, unique=True)
    cin = models.CharField(
        max_length=8,
        validators=[cin_validator],
        null=True,
        unique=True,
        verbose_name="CIN",
        help_text="Carte d'Identité Nationale (8 chiffres)"
    )
    username = models.CharField(max_length=200, null=True, blank=True)
    full_name = models.CharField(max_length=255, blank=True, null=True)
    birth_date = models.DateField(null=True, blank=True, verbose_name="Date de naissance")
    association = models.ForeignKey(
        AssociationAccount,
        on_delete=models.CASCADE,
        related_name='users',
        null=True, blank=True
    )
    # Add role to the user
    role = models.ForeignKey(
        Role,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='users'
    )
    # New fields for user validation
    is_validated = models.BooleanField(default=False, help_text="Indicates if the user has been validated by admin")
    validated_by = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='validated_users'
    )
    validation_date = models.DateTimeField(null=True, blank=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        if self.full_name:
            return f"{self.full_name} ({self.email})"
        return self.email


# Password Reset Signal
@receiver(reset_password_token_created)
def password_reset_token_created(reset_password_token, *args, **kwargs):
    sitelink = "http://localhost:5173/"
    token = "{}".format(reset_password_token.key)
    full_link = str(sitelink) + str("password-reset/") + str(token)

    print(token)
    print(full_link)

    context = {
        'full_link': full_link,
        'email_adress': reset_password_token.user.email,
        'full_name': reset_password_token.user.full_name or reset_password_token.user.email
    }

    html_message = render_to_string("backend/email.html", context=context)
    plain_message = strip_tags(html_message)

    msg = EmailMultiAlternatives(
        subject="Request for resetting password for {title}".format(title=reset_password_token.user.email),
        body=plain_message,
        from_email="sender@example.com",
        to=[reset_password_token.user.email]
    )

    msg.attach_alternative(html_message, "text/html")
    msg.send()
    print("Password reset email sent")