from django.contrib import admin
from django.urls import path, include
from knox import views as knox_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # User URLs
    path('users/', include('users.urls')),

    # API URLs for DRF
    path('api/', include('api.urls')),  # This will use the DRF browsable interface

    # Authentication (Knox)
    path('auth/logout/', knox_views.LogoutView.as_view(), name='knox_logout'),
    path('auth/logoutall/', knox_views.LogoutAllView.as_view(), name='knox_logoutall'),

    # Password reset
    path('auth/password_reset/', include('django_rest_passwordreset.urls', namespace='password_reset')),

    # Chatbot URLs
    path('chatbot/', include('chatbot.urls')),

    # Financial management URLs
    path('finances/', include('finances.urls')),

    # Meetings management URLs
    path('meetings/', include('meetings.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)