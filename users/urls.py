from django.urls import path, include
from .association_management_views import AssociationManagementViewSet
from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.conf.urls.static import static
from .views import *


router = DefaultRouter()
router.register('register', RegisterViewset, basename='register')
router.register('login', LoginViewset, basename='login')
router.register('users', UserViewset, basename='users')
router.register('logout', LogoutViewset, basename='logout')
router.register('register-association', AssociationRegisterViewset, basename='register-association')
router.register('associations', AssociationListViewset, basename='associations')
router.register('profile', UserProfileViewSet, basename='user-profile')
router.register('roles', RoleViewSet, basename='roles')
router.register('association-verification', AssociationAccountViewSet, basename='association-verification')
router.register('association-management', AssociationManagementViewSet, basename='association-management')
urlpatterns = [
    path('', include(router.urls)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)