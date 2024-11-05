import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

app = Flask(__name__)

# Load Kubernetes configuration with a direct path
config.load_kube_config(config_file="~/.kube/config")
v1 = client.CoreV1Api()

class QueryResponse(BaseModel):
    query: str
    answer: str

def generate_kubernetes_command(query):
    """
    Generates a single line of Python code for a Kubernetes command.
    The command must only read data and answer directly without extra details.
    The prompt specifies that we are using a Minikube setup and v1 client.
    """
    prompt = f"""
    You are an AI assistant skilled in Kubernetes and Python. Your task is to generate a single line of Python code
    to answer specific Kubernetes-related questions for a Minikube setup using the Kubernetes client library (v1 client).

    Please carefully consider the question provided in each case. The command you generate should:
    - Only read data (performing a read-only action) without modifying any Kubernetes resources.
    - Store the relevant answer directly in the variable 'result' as a list or string, with no additional metadata or formatting.
    - Avoid using any unique identifiers or unnecessary details in the response. For instance, use concise names like "mongodb" instead of "mongodb-123456".
    - Ensure compatibility with the v1 client in Python and Minikube clusters. 

    Given the question: '{query}', generate a single line of code that:
    - Uses the pre-defined Kubernetes client 'v1'.
    - Answers the question concisely and directly, storing only the required information in the 'result' variable.
    - Does not include any code fences, such as ```python or ```.

    Remember, only use read operations and return a minimal and direct answer.
    """
    
    logging.info(f"Prompt for command generation: {prompt.strip()}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in Kubernetes and Python. Generate minimal, read-only code for specific queries on a Minikube cluster using the v1 client."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    
    command = response.choices[0].message['content'].strip()
    logging.info(f"Generated command: {command}")
    return command

def execute_generated_command(command):
    """
    Executes the generated Kubernetes command and returns the result.
    """
    local_vars = {}
    command = command.replace("```python", "").replace("```", "").strip()
    logging.debug(f"Executing command: {command}")

    try:
        exec(command, globals(), local_vars)
        result = local_vars.get('result', "No result returned")
        logging.debug(f"Execution result: {result}")
        return result
    except Exception as e:
        logging.error(f"Execution error: {str(e)}")
        return f"Error executing command: {str(e)}"

def format_result_with_gpt(query, result):
    """
    Formats the raw result into a concise answer.
    The answer should contain only the direct response without identifiers or dynamic data.
    """
    prompt = f"""
    You are an AI assistant skilled in summarizing technical data. Given the question: '{query}' and the raw result: '{result}',
    provide only the direct answer without any metadata, unique identifiers, or extra formatting. 
    Return only the concise and relevant answer that directly addresses the question."""
    
    logging.debug(f"Prompt for result formatting: {prompt.strip()}")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in summarizing technical data concisely, without extra identifiers."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=20,
        temperature=0.3,
    )
    
    answer = response.choices[0].message['content'].strip()
    logging.debug(f"Formatted answer: {answer}")
    return answer

@app.route('/query', methods=['POST'])
def create_query():
    try:
        request_data = request.json
        query = request_data.get('query')
        
        logging.info(f"Received query: {query}")
        
        # Step 1: Generate Kubernetes command
        command = generate_kubernetes_command(query)
        logging.info(f"Generated command: {command}")
        
        # Step 2: Execute the command
        raw_result = execute_generated_command(command)
        logging.info(f"Raw execution result: {raw_result}")
        
        # Step 3: Format the result
        answer = format_result_with_gpt(query, raw_result)
        logging.info(f"Formatted answer: {answer}")
        
        response = QueryResponse(query=query, answer=answer)
        return jsonify(response.dict())
    
    except ValidationError as e:
        logging.error(f"Validation Error: {e.errors()}")
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Minimal debug route to ensure Kubernetes connectivity remains functional
@app.route('/test_kube_connection', methods=['GET'])
def test_kube_connection():
    try:
        namespaces = [ns.metadata.name for ns in v1.list_namespace().items]
        logging.info("Kubernetes connection successful.")
        return jsonify({"namespaces": namespaces})
    except Exception as e:
        logging.error(f"Kubernetes connection failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)
