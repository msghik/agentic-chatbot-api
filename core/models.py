from django.db import models
from django.utils import timezone

class AgentTool(models.Model):
    METHOD_CHOICES = [
        ("GET", "GET"),
        ("POST", "POST"),
    ]

    name = models.CharField(max_length=100, unique=True, help_text="Identifier (e.g., 'get_weather')")
    description = models.TextField(help_text="Instructions for Gemma 4 on when to use this API.")
    url = models.URLField(help_text="The actual REST API endpoint to call.")
    method = models.CharField(max_length=10, choices=METHOD_CHOICES, default="GET")
    parameters_schema = models.JSONField(default=dict, blank=True, help_text="OpenAPI JSON parameters schema.")
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class AgentAuditLog(models.Model):
    STATUS_CHOICES = [
        ("SUCCESS", "Success"),
        ("ERROR", "Error"),
    ]

    timestamp = models.DateTimeField(default=timezone.now)
    user_prompt = models.TextField(help_text="The exact question the user asked.")
    
    # Tool Execution Data
    tool_called = models.CharField(max_length=100, blank=True, null=True, help_text="Which database tool was triggered?")
    tool_arguments = models.JSONField(blank=True, null=True, help_text="The exact arguments Gemma passed to the tool.")
    tool_raw_response = models.JSONField(blank=True, null=True, help_text="The raw JSON data returned by the remote API.")
    
    # Final Output
    final_response = models.TextField(blank=True, null=True, help_text="The final textual response sent to the user.")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="SUCCESS")
    error_message = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"[{self.status}] {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Tool: {self.tool_called or 'None'}"
