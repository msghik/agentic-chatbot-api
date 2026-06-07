# core/services.py
import os
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
from .models import AgentTool, AgentAuditLog

def run_django_agent(user_prompt: str) -> dict:
    """Returns a dictionary containing the response text and the audit log ID."""
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
        system_instruction="You are an autonomous API agent helper. Route tools efficiently.",
        tools=gemma_tools,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="HIGH")
    )

    # 2. Call the Model
    try:
        response = client.models.generate_content(
            model="gemma-4-31b-it", contents=user_prompt, config=config
        )
    except Exception as e:
        audit.status = "ERROR"
        audit.error_message = f"Gemma API Error: {str(e)}"
        audit.save()
        return {"response": "⚠️ Service temporarily unavailable.", "log_id": audit.id}

    # 3. Handle Tool Executions
    if response.function_calls:
        # For simplicity, we track the first tool call in the audit log
        call = response.function_calls[0]
        clean_args = dict(call.args)
        
        # Typosquatting fix
        if "ongitude" in clean_args and "longitude" not in clean_args:
            clean_args["longitude"] = clean_args.pop("ongitude")
            
        # Update Audit Log with Intent
        audit.tool_called = call.name
        audit.tool_arguments = clean_args
        
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
            audit.error_message = str(e)
            
        audit.tool_raw_response = api_data

        # Final turn back to Gemma
        try:
            final_turn = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
                    response.candidates[0].content,
                    types.Content(role="function", parts=[
                        types.Part.from_function_response(name=call.name, response={"result": api_data})
                    ])
                ]
            )
            audit.final_response = final_turn.text
            audit.save()
            return {"response": final_turn.text, "log_id": audit.id}
            
        except Exception as e:
            audit.status = "ERROR"
            audit.error_message = f"Final Turn Error: {str(e)}"
            audit.save()
            return {"response": "⚠️ Failed to finalize explanation.", "log_id": audit.id}

    # 4. Standard Text Response (No tools triggered)
    audit.final_response = response.text
    audit.save()
    return {"response": response.text, "log_id": audit.id}
