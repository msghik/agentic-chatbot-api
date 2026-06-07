# core/urls.py
from django.urls import path
from .views import AgentRunAPIView, AuditLogListAPIView

urlpatterns = [
    path('run/', AgentRunAPIView.as_view(), name='agent-run'),
    path('logs/', AuditLogListAPIView.as_view(), name='agent-logs'),
]
