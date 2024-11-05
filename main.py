import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

# Check for and import each required module, logging any errors
missing_modules = []

# Try importing each library with exception handling
try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    missing_modules.append("pydantic")

try:
    from kubernetes import client, config
except ImportError:
    missing_modules.append("kubernetes")

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    missing_modules.append("openai")

# If there are missing modules, log and notify the user
if missing_modules:
    missing_message = f"Missing required modules: {', '.join(missing_modules)}"
    logging.error(missing_message)
    raise ImportError(missing_message)

# Continue with the rest of the application setup
app = Flask(__name__)

# Load Kubernetes configuration
try:
    config.load_kube_config(config_file="~/.kube/config")
    v1 = client.CoreV1Api()
    logging.info("Kubernetes configuration loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Kubernetes configuration: {str(e)}")
    v1 = None

# Define the response model
class QueryResponse(BaseModel):
    query: str
    answer: str

def generate_kubernetes_command(query):
    """
    Generates a Kubernetes command for a given query using GPT-4.
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
    try:
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
    except Exception as e:
        logging.error(f"Error generating Kubernetes command: {str(e)}")
        return None

def execute_generated_command(command):
    """
    Executes the generated Kubernetes command and returns the result.
    """
    if not command:
        logging.error("No command to execute.")
        return "No command generated."

    local_vars = {}
    command = command.replace("```python", "").replace("```", "").strip()
    logging.debug(f"Executing command: {command}")

    try:
        exec(command, globals(), local_vars)
        result = local_vars.get('result', "No result returned")
        logging.debug(f"Execution result: {result}")
        return result
    except AttributeError as e:
        logging.error(f"Attribute error during command execution: {str(e)}")
        return "Kubernetes client method not supported on Minikube."
    except Exception as e:
        logging.error(f"Execution error: {str(e)}")
        return f"Error executing command: {str(e)}"

def format_result_with_gpt(query, result):
    """
    Formats the raw result into a concise answer.
    """
    prompt = f"""
    You are an AI assistant skilled in summarizing technical data. Given the question: '{query}' and the raw result: '{result}',
    provide only the direct answer without any metadata, unique identifiers, or extra formatting. For example, return 'mongodb' instead of 'mongodb-56c598c8fc'.
    Return only the concise and relevant answer that directly addresses the question.
    """

    logging.debug(f"Prompt for result formatting: {prompt.strip()}")
    try:
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
    except Exception as e:
        logging.error(f"Error formatting result with GPT: {str(e)}")
        return "Error formatting answer."

@app.route('/query', methods=['POST'])
def create_query():
    try:
        request_data = request.json
        query = request_data.get('query')

        if not query:
            logging.error("No query provided in request.")
            return jsonify({"error": "No query provided"}), 400

        logging.info(f"Received query: {query}")

        # Step 1: Generate Kubernetes command
        command = generate_kubernetes_command(query)
        if not command:
            logging.error("Failed to generate command.")
            return jsonify({"error": "Failed to generate command"}), 500

        # Step 2: Execute the command
        raw_result = execute_generated_command(command)
        if "Error" in raw_result:
            logging.error("Error in command execution.")
            return jsonify({"error": raw_result}), 500

        # Step 3: Format the result
        answer = format_result_with_gpt(query, raw_result)
        if "Error" in answer:
            logging.error("Error in formatting result.")
            return jsonify({"error": answer}), 500

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
    if not v1:
        logging.error("Kubernetes client not initialized.")
        return jsonify({"error": "Kubernetes client not initialized"}), 500

    try:
        namespaces = [ns.metadata.name for ns in v1.list_namespace().items]
        logging.info("Kubernetes connection successful.")
        return jsonify({"namespaces": namespaces})
    except Exception as e:
        logging.error(f"Kubernetes connection failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)
