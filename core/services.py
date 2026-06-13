# core/services.py
import os
import requests
import json
from .models import AgentTool, AgentAuditLog

def run_django_agent(user_prompt: str) -> dict:
    """
    Orchestrates the dynamic agent loop. It fetches active tools from the database,
    proposes them to Gemma 4, intercepts any tool calls, handles arguments safely,
    executes the target API locally, and feeds results back for a final explanation.
    Returns a dictionary containing the response text and the audit log ID.
    """
    # 1. Initialize the Audit Log Session
    audit = AgentAuditLog.objects.create(user_prompt=user_prompt)

    db_tools = AgentTool.objects.filter(is_active=True)

    # Format tools for LM Studio payload
    tools = []
    for t in db_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema
            }
        })

    lm_studio_url = "http://localhost:1234/api/v1/chat"
    system_prompt = (
        "You are an autonomous API agent helper. Analyze the user request. "
        "If an applicable database tool matches the intent, call it with appropriate arguments. "
        "Once you receive the structural results, translate them back into an elegant, conversational breakdown."
    )

    payload = {
        "model": "gemma-4-e2b-it",
        "system_prompt": system_prompt,
        "input": user_prompt,
    }
    if tools:
        payload["tools"] = tools

    # 2. First Execution Turn (Intent Analysis & Tool Routing)
    try:
        res = requests.post(
            lm_studio_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        res.raise_for_status()
        response_data = res.json()
    except Exception as e:
        audit.status = "ERROR"
        audit.error_message = f"LM Studio API Turn 1 Error: {str(e)}"
        audit.save()
        return {"response": "⚠️ Service temporarily unavailable.", "log_id": audit.id}

    # 3. Handle Tool Execution Layer if Gemma decides to call an API

    # Try different known tool call locations
    function_calls = response_data.get("function_calls", [])
    if not function_calls and "tool_calls" in response_data:
        function_calls = response_data["tool_calls"]
    if not function_calls and "choices" in response_data and len(response_data["choices"]) > 0:
        message = response_data["choices"][0].get("message", {})
        if "tool_calls" in message:
            function_calls = message["tool_calls"]
        elif "function_calls" in message:
            function_calls = message["function_calls"]

    if function_calls:
        # For simplicity, we track the first tool call in the audit log
        call = function_calls[0]

        call_name = ""
        clean_args = {}

        if "function" in call:
            call_name = call["function"]["name"]
            try:
                args_str = call["function"].get("arguments", "{}")
                if isinstance(args_str, dict):
                    clean_args = args_str
                else:
                    clean_args = json.loads(args_str)
            except json.JSONDecodeError:
                clean_args = {}
        else:
            call_name = call.get("name", "")
            clean_args = call.get("args", {})
            if isinstance(clean_args, str):
                try:
                    clean_args = json.loads(clean_args)
                except json.JSONDecodeError:
                    clean_args = {}

        # Typosquatting fix: Handle common dropped leading characters
        if "ongitude" in clean_args and "longitude" not in clean_args:
            clean_args["longitude"] = clean_args.pop("ongitude")

        # Update Audit Log with Intent
        audit.tool_called = call_name
        audit.tool_arguments = clean_args

        # Look up the definition directly in the database
        try:
            target_tool = AgentTool.objects.get(name=call_name)
            
            # PATH-ROUTING FIX: If the URL ends with a trailing slash and we have a single argument,
            # append it to the path instead of sending it as a query parameter (ideal for REST Countries)
            execution_url = target_tool.url
            request_args = clean_args.copy()
            
            if execution_url.endswith("/") and len(request_args) == 1:
                val = list(request_args.values())[0]
                execution_url = f"{execution_url}{val}"
                request_args = {} # Clear query args since it's now in the path

            if target_tool.method == "GET":
                api_res = requests.get(execution_url, params=request_args, timeout=10)
            else:
                api_res = requests.post(execution_url, json=request_args, timeout=10)

            api_res.raise_for_status()
            api_data = api_res.json()
            
        except Exception as e:
            api_data = {"error": f"API Request Failed: {str(e)}"}
            audit.status = "ERROR"
            audit.error_message = f"Tool Execution Failure: {str(e)}"

        audit.tool_raw_response = api_data

        # 4. Second Execution Turn (Feeding data payload back to the model for translation)
        try:
            # We append the tool execution result to the user prompt directly
            second_turn_prompt = (
                f"Original Request: {user_prompt}\n"
                f"Tool Used: {call_name}\n"
                f"Tool Result: {json.dumps(api_data)}\n"
                "Please provide a final conversational breakdown based on these results."
            )

            payload_turn2 = {
                "model": "gemma-4-e2b-it",
                "system_prompt": system_prompt,
                "input": second_turn_prompt,
            }
            if tools:
                payload_turn2["tools"] = tools

            res2 = requests.post(
                lm_studio_url,
                headers={"Content-Type": "application/json"},
                json=payload_turn2,
                timeout=10
            )
            res2.raise_for_status()
            final_turn_data = res2.json()

            final_text = ""
            if "response" in final_turn_data:
                final_text = final_turn_data["response"]
            elif "choices" in final_turn_data and len(final_turn_data["choices"]) > 0:
                final_text = final_turn_data["choices"][0].get("message", {}).get("content", "")
            else:
                final_text = final_turn_data.get("text", "")

            audit.final_response = final_text
            # If an upstream API request failed earlier, keep the status as ERROR but save the text explanation
            if audit.status != "ERROR":
                audit.status = "SUCCESS"
            audit.save()
            return {"response": final_text, "log_id": audit.id}

        except Exception as e:
            audit.status = "ERROR"
            audit.error_message = f"Final Turn Error: {str(e)}"
            audit.save()
            return {"response": "⚠️ Failed to finalize explanation.", "log_id": audit.id}

    # 5. Standard Text Response (No tools triggered)
    final_text = ""
    if "response" in response_data:
        final_text = response_data["response"]
    elif "choices" in response_data and len(response_data["choices"]) > 0:
        final_text = response_data["choices"][0].get("message", {}).get("content", "")
    else:
        final_text = response_data.get("text", "")

    audit.final_response = final_text
    audit.save()
    return {"response": final_text, "log_id": audit.id}
