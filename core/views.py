# core/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import run_django_agent
from .models import AgentAuditLog

class AgentRunAPIView(APIView):
    """
    POST /api/agent/run/
    Accepts: {"prompt": "What is the weather in Austin?"}
    """
    def post(self, request):
        prompt = request.data.get("prompt")
        
        if not prompt:
            return Response({"error": "The 'prompt' field is required."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Execute the audited agent loop
        result = run_django_agent(prompt)
        
        # Return the payload to the frontend
        return Response({
            "prompt": prompt,
            "agent_response": result["response"],
            "audit_log_id": result["log_id"]
        }, status=status.HTTP_200_OK)

class AuditLogListAPIView(APIView):
    """
    GET /api/agent/logs/
    Returns a quick history of all agent actions.
    """
    def get(self, request):
        logs = AgentAuditLog.objects.all().order_by('-timestamp')[:50]
        data = []
        for log in logs:
            data.append({
                "id": log.id,
                "timestamp": log.timestamp,
                "prompt": log.user_prompt,
                "tool_called": log.tool_called,
                "status": log.status
            })
        return Response(data, status=status.HTTP_200_OK)
