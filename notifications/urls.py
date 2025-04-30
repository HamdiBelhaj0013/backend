# notifications/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import NotificationViewSet, OfficialLetterLogViewSet


# Create a router and register our viewsets with it
router = DefaultRouter()
router.register('notifications', NotificationViewSet, basename='notification')
router.register('letters', OfficialLetterLogViewSet, basename='letter')

urlpatterns = [
    path('', include(router.urls)),

]