import os
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
from .models import AgentTool

def run_django_agent(user_prompt: str) -> str:
    """
    Orchestrates the dynamic agent loop. It fetches active tools from the database,
    proposes them to Gemma 4, intercepts any tool calls, handles arguments safely,
    executes the target API locally, and feeds results back for a final explanation.
    """
    # 1. Initialize the Google GenAI Client
    client = genai.Client(api_key="AIzaSyDFvcFAe42jn5PjypzidtuVzSKRrgd9dyg")

    # 2. Gather active tools from your SQLite database
    db_tools = AgentTool.objects.filter(is_active=True)
    
    declarations = []
    for db_tool in db_tools:
        func_decl = types.FunctionDeclaration(
            name=db_tool.name,
            description=db_tool.description,
            parameters_json_schema=db_tool.parameters_schema
        )
        declarations.append(func_decl)

    gemma_tools = [types.Tool(function_declarations=declarations)] if declarations else None

    # 3. Configure Gemma 4 with High Reasoning/Thinking level enabled
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

    # 4. First Execution Turn (Intent Analysis & Tool Routing)
    try:
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=user_prompt,
            config=config
        )
    except APIError as e:
        return f"⚠️ Google GenAI Server Error: {e.message}. The endpoint had a transient hiccup, please run it again."
    except Exception as e:
        return f"⚠️ Unexpected Error communicating with Gemma: {str(e)}"

    # 5. Handle Tool Execution Layer if Gemma decides to call an API
    if response.function_calls:
        for call in response.function_calls:
            # Clone args into a mutable dictionary to normalize variations
            clean_args = dict(call.args)
            
            # Auto-Correction Layer: Capture common parameter spelling variations or drops
            if "ongitude" in clean_args and "longitude" not in clean_args:
                clean_args["longitude"] = clean_args.pop("ongitude")
                
            print(f"⚙️ [Executing DB Tool]: {call.name} with processed args: {clean_args}")
            
            # Look up the definition directly in the database
            try:
                target_tool = AgentTool.objects.get(name=call.name)
            except AgentTool.DoesNotExist:
                api_data = {"error": f"Tool '{call.name}' is no longer active or missing in the database."}
                return f"⚠️ Agent structural failure: requested non-existent tool '{call.name}'."

            # Make the live network call safely with a strict timeout window
            try:
                if target_tool.method == "GET":
                    api_response = requests.get(target_tool.url, params=clean_args, timeout=10)
                else:
                    api_response = requests.post(target_tool.url, json=clean_args, timeout=10)
                    
                api_response.raise_for_status()
                api_data = api_response.json()
            except requests.exceptions.RequestException as e:
                api_data = {"error": f"Failed to successfully reach external remote service: {str(e)}"}
            
            # 6. Second Execution Turn (Feeding data payload back to the model for translation)
            try:
                final_turn = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
                        response.candidates[0].content, # Historical context of the original function request
                        types.Content(role="function", parts=[
                            types.Part.from_function_response(name=call.name, response={"result": api_data})
                        ])
                    ]
                )
                return final_turn.text
            except APIError as e:
                return f"⚠️ Google GenAI Turn 2 Server Error: {e.message}. Please re-submit your question."
            except Exception as e:
                return f"⚠️ Failed to synthesize final response block: {str(e)}"
                
    # If no tool calls were generated, return the baseline conversational output text
    return response.text
