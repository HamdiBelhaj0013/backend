from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    MeetingViewSet,
    MeetingAttendeeViewSet,
    MeetingAgendaItemViewSet,
    MeetingReportViewSet,
    MeetingNotificationViewSet
)

router = DefaultRouter()
router.register('meetings', MeetingViewSet, basename='meeting')
router.register('attendees', MeetingAttendeeViewSet, basename='meeting-attendee')
router.register('agenda-items', MeetingAgendaItemViewSet, basename='meeting-agenda-item')
router.register('reports', MeetingReportViewSet, basename='meeting-report')
router.register('notifications', MeetingNotificationViewSet, basename='meeting-notification')

urlpatterns = [
    path('', include(router.urls)),

    # Add these custom routes for meeting responses
    path('attendees/<int:pk>/response/<str:token>/',
         MeetingAttendeeViewSet.as_view({'get': 'respond_with_token', 'post': 'respond_with_token'}),
         name='attendee-respond-with-token'),

    path('meetings/<int:pk>/respond/',
         MeetingViewSet.as_view({'post': 'respond_to_invitation'}),
         name='meeting-respond'),
]