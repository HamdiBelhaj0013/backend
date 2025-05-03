# myapp/context_processors.py

def base_url(request):
    """
    Context processor to add the base URL to templates.
    This is used in email templates for generating full URLs.
    """
    from django.conf import settings

    # Use the configured EMAIL_BASE_URL from settings
    # Fall back to default value if not configured
    base_url = getattr(settings, 'EMAIL_BASE_URL', 'http://localhost:5173')

    return {'base_url': base_url}