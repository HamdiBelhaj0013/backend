from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DonorViewSet, TransactionViewSet, BudgetAllocationViewSet,
    FinancialReportViewSet, FinancialDashboardViewSet
)

router = DefaultRouter()
router.register('donors', DonorViewSet, basename='donor')
router.register('transactions', TransactionViewSet, basename='transaction')
router.register('budget-allocations', BudgetAllocationViewSet, basename='budget-allocation')
router.register('financial-reports', FinancialReportViewSet, basename='financial-report')
router.register('dashboard', FinancialDashboardViewSet, basename='financial-dashboard')


urlpatterns = [
    path('', include(router.urls)),
]