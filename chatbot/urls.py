from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ConversationViewSet, DocumentViewSet

router = DefaultRouter()
router.register(r'conversations', ConversationViewSet, basename='conversation')
router.register(r'documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('', include(router.urls)),
]