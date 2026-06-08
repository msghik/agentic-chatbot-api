# core/services.py
import os
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
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

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    db_tools = AgentTool.objects.filter(is_active=True)

    declarations = [
        types.FunctionDeclaration(
            name=t.name, description=t.description, parameters_json_schema=t.parameters_schema
        ) for t in db_tools
    ]
    gemma_tools = [types.Tool(function_declarations=declarations)] if declarations else None

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are an autonomous API agent helper. Analyze the user request. "
            "If an applicable database tool matches the intent, call it with appropriate arguments. "
            "Once you receive the structural results, translate them back into an elegant, conversational breakdown."
        ),
        tools=gemma_tools,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="HIGH")
    )

    # 2. First Execution Turn (Intent Analysis & Tool Routing)
    try:
        response = client.models.generate_content(
            model="gemma-4-31b-it", contents=user_prompt, config=config
        )
    except Exception as e:
        audit.status = "ERROR"
        audit.error_message = f"Gemma API Turn 1 Error: {str(e)}"
        audit.save()
        return {"response": "⚠️ Service temporarily unavailable.", "log_id": audit.id}

    # 3. Handle Tool Execution Layer if Gemma decides to call an API
    if response.function_calls:
        # For simplicity, we track the first tool call in the audit log
        call = response.function_calls[0]
        clean_args = dict(call.args)

        # Typosquatting fix: Handle common dropped leading characters
        if "ongitude" in clean_args and "longitude" not in clean_args:
            clean_args["longitude"] = clean_args.pop("ongitude")

        # Update Audit Log with Intent
        audit.tool_called = call.name
        audit.tool_arguments = clean_args

        # Look up the definition directly in the database
        try:
            target_tool = AgentTool.objects.get(name=call.name)
            if target_tool.method == "GET":
                api_res = requests.get(target_tool.url, params=clean_args, timeout=10)
            else:
                api_res = requests.post(target_tool.url, json=clean_args, timeout=10)

            api_res.raise_for_status()
            api_data = api_res.json()
        except Exception as e:
            api_data = {"error": f"API Request Failed: {str(e)}"}
            audit.status = "ERROR"
            audit.error_message = f"Tool Execution Failure: {str(e)}"

        audit.tool_raw_response = api_data

        # 4. Second Execution Turn (Feeding data payload back to the model for translation)
        try:
            final_turn = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
                    response.candidates[0].content,  # Historical context of original tool request
                    types.Content(role="user", parts=[
                        types.Part.from_function_response(
                            name=call.name, 
                            response={"result": api_data}
                        )
                    ])
                ],
                config=config  # Re-inject instructions and tools so Turn 2 parsing succeeds
            )
            audit.final_response = final_turn.text
            # If an upstream API request failed earlier, keep the status as ERROR but save the text explanation
            if audit.status != "ERROR":
                audit.status = "SUCCESS"
            audit.save()
            return {"response": final_turn.text, "log_id": audit.id}

        except Exception as e:
            audit.status = "ERROR"
            audit.error_message = f"Final Turn Error: {str(e)}"
            audit.save()
            return {"response": "⚠️ Failed to finalize explanation.", "log_id": audit.id}

    # 5. Standard Text Response (No tools triggered)
    audit.final_response = response.text
    audit.save()
    return {"response": response.text, "log_id": audit.id}
