from django.urls import path, include, re_path
from rest_framework.routers import DefaultRouter
from .views import (
    MeetingViewSet,
    MeetingAttendeeViewSet,
    MeetingReportViewSet,
    MeetingNotificationViewSet
)

# Create a router for standard viewsets
router = DefaultRouter()
router.register(r'meetings', MeetingViewSet, basename='meeting')
router.register(r'attendees', MeetingAttendeeViewSet, basename='meeting-attendee')
router.register(r'reports', MeetingReportViewSet, basename='meeting-report')
router.register(r'notifications', MeetingNotificationViewSet, basename='meeting-notification')

# Define URL patterns
urlpatterns = [
    # IMPORTANT: Define custom routes BEFORE including router.urls

    # Meeting response with direct yes/no/maybe responses
    path('response/<int:pk>/<str:token>/<str:response_type>/',
         MeetingAttendeeViewSet.as_view({'get': 'respond_with_type', 'post': 'respond_with_type'}),
         name='attendee-respond-with-type'),

    # Meeting response without direct response
    path('response/<int:pk>/<str:token>/',
         MeetingAttendeeViewSet.as_view({'get': 'respond_with_type', 'post': 'respond_with_type'}),
         name='attendee-respond-without-type'),

    # Include router URLs
    path('', include(router.urls)),

    # Meeting invitation response
    path('meetings/<int:pk>/respond/',
         MeetingNotificationViewSet.as_view({'post': 'respond_to_invitation'}),
         name='meeting-respond'),
]