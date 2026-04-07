import asyncio
import os
import json
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llama_cpp import Llama
from datetime import datetime

base_path = os.path.dirname(__file__)
# Create a timestamped folder and log file
log_base = os.path.join(base_path, "logs")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_base, f"session_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "dialogue_log.txt")

# for logging user input and responses
def log_to_file(text):
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")

# === System Prompt ===
SYSTEM_PROMPT = """
    You are an XAI assistant. Decide which tools to call, in what order. Keep answers concise unless the user asks for depth. 
    When you use tools, explain their results in plain language and cite which tool you used (e.g., "(from SHAP)").
    If the user asks "why", prefer SHAP. If they ask "how to change outcome", prefer counterfactuals. 
    If they ask about meanings, use knowledge. If they ask for a prediction, call model.predict.
"""

# === Server configurations ===
server_configs = {
    "xai": "servers/xai_server.py",
    "knowledge": "servers/knowledge_server.py",
    "logger": "servers/logger_server.py",
}

# === Configure Gemini client ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# Evaluate the response
def evaluate_response_with_judge(prompt, response, user_input,
                                 judge_model_path="/Users/arijeet/Downloads/llama-pro-8b-instruct.Q3_K_L.gguf"):
    judge = Llama(model_path=judge_model_path)

    eval_prompt = f"""
        ### Instruction:
        You are an expert AI tasked with evaluating assistant responses.

        Rate the assistant's explanation based on:
        1. Clarity
        2. Correctness (logical and factual alignment with question)
        3. Helpfulness (how well it answers the user's intent)

        Provide a score and short justification.

        ### User Question:
        {user_input}

        ### Assistant's Response:
        {response}

        ### Evaluation Format:
        Rating: [Good | Acceptable | Needs Improvement]
        Reason: <Your short explanation here>

        ### Evaluation:
    """

    result = judge(eval_prompt, max_tokens=128, stop=["###", "\n\n"])
    return result["choices"][0]["text"].strip()

def extract_json(response: str) -> str:
    """
    Remove Markdown fences (```json ... ``` or ``` ... ```).
    """
    response = response.strip()
    if response.startswith("```"):
        # Remove the first and last fenced lines
        lines = response.splitlines()
        # drop the first line (``` or ```json) and last line (```)
        response = "\n".join(lines[1:-1])
    return response

def llm_client(message: str):
    """
    Send a message to Gemini and return the response.
    """
    response = model.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [f"{SYSTEM_PROMPT}\n\nUser query: {message}"]
            }
        ]
    )
    return response.text.strip() if response.text else ""

def get_prompt_to_identify_tool_and_arguments(query, tools):
    tools_description = "\n".join(
        [f"- {tool.name}, {tool.description}, schema: {json.dumps(tool.inputSchema)}"
         for tool in tools]
    )

    return (
        f"You are a helpful assistant with access to these tools:\n\n"
        f"{tools_description}\n\n"
        f"User's Question: {query}\n\n"
        "If no tool is needed, reply directly.\n\n"
        "IMPORTANT: When you need to use a tool, you must provide ALL required arguments "
        "according to the tool's schema. Respond ONLY with this JSON format:\n"
        "{\n"
        '    "tool": "tool-name",\n'
        '    "arguments": {\n'
        '        "argument-name": "value"\n'
        "    }\n"
        "}\n"
    )

async def run_dialogue(server_key: str):
    log_to_file("=== NEW DIALOGUE SESSION STARTED ===")
    log_to_file(f"Timestamp: {timestamp}\n")
    if server_key not in server_configs:
        raise ValueError(f"Unknown server key: {server_key}")

    server_params = StdioServerParameters(
        command="python", args=[server_configs[server_key]]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools from {server_key}: {tools}")

            # === Dialogue loop ===
            while True:
                query = input("\nUser > ").strip()
                if query.lower() in ["exit", "quit"]:
                    print("Exiting dialogue.")
                    break

                prompt = get_prompt_to_identify_tool_and_arguments(query, tools.tools)
                llm_response = llm_client(prompt)
                print(f"LLM response : {llm_response}")
                log_to_file(f"LLM response : {llm_response}")

                cleaned = extract_json(llm_response)
                try:
                    # Try to parse JSON tool call
                    tool_call = json.loads(cleaned)
                    print(f"Tool call: {tool_call}")
                    log_to_file(f"Tool call: {tool_call}")

                    if "tool" in tool_call and "arguments" in tool_call:
                        result = await session.call_tool(
                            tool_call["tool"], arguments=tool_call["arguments"]
                        )

                        # Convert tool result into plain language with LLM
                        if result.content and result.content[0].text:
                            raw_output = result.content[0].text

                            # Use LLM to rewrite into human-readable explanation
                            explanation_prompt = f"""
                            The user asked: {query}
                            The tool {tool_call['tool']} returned the following result: {raw_output}

                            Please explain this result to the user in clear, natural language.
                            """

                            friendly_response = model.generate_content(explanation_prompt).text

                            # Save to variable for later use
                            final_response = friendly_response.strip()

                            # Show to user
                            print(f"Assistant > {final_response}")
                            log_to_file(f"Assistant > {final_response}")
                            # eval_feedback = evaluate_response_with_judge(explanation_prompt, final_response, query)
                            # print(f"\n[Judge Evaluation]: {eval_feedback}")
                        else:
                            print(f"Assistant > Tool {tool_call['tool']} executed but returned no text.")

                        continue  # handled, go to next turn

                except json.JSONDecodeError:
                    pass  # Not JSON, fall back to direct text

                # If not a JSON tool call → treat as direct text response
                print(f"Assistant > {llm_response}")


if __name__ == "__main__":
    # Start conversation with the "xai" server by default
    asyncio.run(run_dialogue("xai"))
