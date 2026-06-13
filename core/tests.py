from django.test import TestCase
from unittest.mock import patch, MagicMock
from .models import AgentTool, AgentAuditLog
from .services import run_django_agent

class TestDjangoAgent(TestCase):
    def setUp(self):
        # Create a sample tool
        self.tool = AgentTool.objects.create(
            name="get_weather",
            description="Get the weather for a location",
            url="http://mock-weather-api/weather",
            method="GET",
            is_active=True,
            parameters_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        )

    @patch('core.services.requests.post')
    def test_standard_text_response(self, mock_post):
        # Mock LM Studio response with no tool calls
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Here is a standard response."
        }
        mock_post.return_value = mock_response

        result = run_django_agent("Say hello")

        self.assertEqual(result["response"], "Here is a standard response.")

        # Verify audit log
        audit = AgentAuditLog.objects.get(id=result["log_id"])
        self.assertEqual(audit.final_response, "Here is a standard response.")
        self.assertIsNone(audit.tool_called)

    @patch('core.services.requests.get')
    @patch('core.services.requests.post')
    def test_tool_call_execution(self, mock_post, mock_get):
        # 1. Mock Turn 1 LM Studio response (Tool Call)
        mock_post_response_turn1 = MagicMock()
        mock_post_response_turn1.json.return_value = {
            "function_calls": [
                {
                    "name": "get_weather",
                    "args": {"location": "London"}
                }
            ]
        }

        # 2. Mock Turn 2 LM Studio response (Final text)
        mock_post_response_turn2 = MagicMock()
        mock_post_response_turn2.json.return_value = {
            "response": "The weather in London is sunny."
        }

        # Setup side_effect for post to return turn 1 then turn 2
        mock_post.side_effect = [mock_post_response_turn1, mock_post_response_turn2]

        # 3. Mock the Tool API response
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"weather": "sunny"}
        mock_get.return_value = mock_get_response

        # Execute
        result = run_django_agent("What is the weather in London?")

        self.assertEqual(result["response"], "The weather in London is sunny.")

        # Verify audit log
        audit = AgentAuditLog.objects.get(id=result["log_id"])
        self.assertEqual(audit.tool_called, "get_weather")
        self.assertEqual(audit.tool_arguments, {"location": "London"})
        self.assertEqual(audit.tool_raw_response, {"weather": "sunny"})
        self.assertEqual(audit.status, "SUCCESS")
