from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    # ViewSets
    ConversationViewSet,
    DocumentViewSet,

    # API Views
    DirectChatView,
    ConversationListView,
    ConversationDetailView,
    ConversationChatView,
    ConversationFeedbackView,
    ChatbotSettingsView,
    DocumentSearchView,
    DocumentDetailView,
)

# Create a router for ViewSets
router = DefaultRouter()
router.register(r'conversations', ConversationViewSet, basename='conversation-viewset')
router.register(r'documents', DocumentViewSet, basename='document')

urlpatterns = [
    # ViewSet routes - for more complex applications
    path('api/', include(router.urls)),

    # Simple view routes - easier to understand and use
    path('conversations/', ConversationListView.as_view(), name='conversation-list'),
    path('conversations/<int:pk>/', ConversationDetailView.as_view(), name='conversation-detail'),
    path('conversations/<int:pk>/chat/', ConversationChatView.as_view(), name='conversation-chat'),
    path('conversations/<int:pk>/feedback/', ConversationFeedbackView.as_view(), name='conversation-feedback'),
    path('settings/', ChatbotSettingsView.as_view(), name='chatbot-settings'),

    # Document routes
    path('documents/search/', DocumentSearchView.as_view(), name='document-search'),
    path('documents/<int:pk>/', DocumentDetailView.as_view(), name='document-detail'),

    # Direct chat endpoint
    path('direct-chat/', DirectChatView.as_view(), name='direct-chat'),

    # Additional functionality
    # Analysis and insights endpoint
    path('conversations/<int:pk>/analyze/',
         ConversationViewSet.as_view({'get': 'analyze'}, name='conversation-analyze')),

    # Regenerate response endpoint
    path('conversations/<int:pk>/regenerate/',
         ConversationViewSet.as_view({'post': 'regenerate'}, name='conversation-regenerate')),
]

# For ease of development, add aliases for the ViewSet detail routes
conversation_detail = ConversationViewSet.as_view({
    'get': 'retrieve',
    'put': 'update',
    'patch': 'partial_update',
    'delete': 'destroy'
})

urlpatterns += [
    # Extra ViewSet actions
    path('api/conversations/<int:pk>/archive/',
         ConversationViewSet.as_view({'post': 'archive'}, name='conversation-archive')),
    path('api/conversations/<int:pk>/restore/',
         ConversationViewSet.as_view({'post': 'restore'}, name='conversation-restore')),
    path('api/conversations/<int:pk>/title/',
         ConversationViewSet.as_view({'post': 'title'}, name='conversation-title')),

    # Document processing routes
    path('api/documents/<int:pk>/process/',
         DocumentViewSet.as_view({'post': 'process'}, name='document-process')),
    path('api/documents/<int:pk>/chunks/',
         DocumentViewSet.as_view({'get': 'chunks'}, name='document-chunks')),
    path('api/documents/<int:pk>/structure/',
         DocumentViewSet.as_view({'get': 'structure'}, name='document-structure')),
]