from rest_framework.routers import DefaultRouter
from .views import MemberViewset, ProjectViewset

router = DefaultRouter()
router.register('project', ProjectViewset, basename='project')
router.register('member', MemberViewset, basename='member')

urlpatterns = router.urls
