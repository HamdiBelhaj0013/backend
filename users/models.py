from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.dispatch import receiver
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags
from django_rest_passwordreset.signals import reset_password_token_created



class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, association=None, **extra_fields):
        if not email:
            raise ValueError('Email is a required field')

        if not association:
            raise ValueError('Association is a required field')  # Ensuring association is provided

        email = self.normalize_email(email)
        extra_fields.setdefault('is_active', True)

        user = self.model(email=email, association=association, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, association=None, **extra_fields):
        if not association:
            raise ValueError('Association is a required field')  # Ensuring association is provided

        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, association, **extra_fields)


# Association Account Model (Main User)
class AssociationAccount(models.Model):
    name = models.CharField(max_length=255, unique=True)
    email = models.EmailField(unique=True)
    cin_recto = models.FileField(upload_to='documents/cin/', blank=True, null=True)
    cin_verso = models.FileField(upload_to='documents/cin/', blank=True, null=True)
    matricule_fiscal = models.CharField(max_length=100, unique=True)
    rne_document = models.FileField(upload_to='documents/rne/', blank=True, null=True)

    def __str__(self):
        return self.name


# Custom User Model
class CustomUser(AbstractUser):
    email = models.EmailField(max_length=200, unique=True)
    username = models.CharField(max_length=200, null=True, blank=True)
    association = models.ForeignKey(
        AssociationAccount,
        on_delete=models.CASCADE,
        related_name='users',
        null=True, blank=False  # Now association is mandatory
    )

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['association']  # Ensuring association is required

    def __str__(self):
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
        'email_adress': reset_password_token.user.email
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
