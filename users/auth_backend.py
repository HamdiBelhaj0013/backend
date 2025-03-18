from django.contrib.auth import get_user_model
User = get_user_model()

class EmailAuthBackend:
    def authenticate(self, request, email=None, password=None):
        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                return user
            return None  # Explicitly return None if password check fails
        except User.DoesNotExist:
            return None
        except Exception as e:
            print(f"Authentication error: {e}")  # Or use logging
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
