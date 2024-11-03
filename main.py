import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config
import openai

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')

app = Flask(__name__)



# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()

class QueryResponse(BaseModel):
    query: str
    answer: str

def generate_kubernetes_command(query):
    """
    Generates a single line of Python code for Kubernetes command.
    """
    prompt = f"""
    Given the question: '{query}', generate a single line of Python code using the Kubernetes client library.
    This line should answer the question by storing only the names of the pods in the 'result' variable as a list of strings.
    Use the pre-defined Kubernetes client 'v1' and make sure the command performs only a read action.
    """
    
    logging.info(f"Prompt for command generation: {prompt.strip()}")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in Kubernetes and Python. Generate minimal code for read-only queries."},
            {"role": "user", "content": prompt.strip()}
        ],
        max_tokens=100,
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
    """
    prompt = f"Given the question: '{query}' and the raw result: '{result}', provide a direct answer only."
    logging.debug(f"Prompt for result formatting: {prompt.strip()}")

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in summarizing technical data."},
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
        
        # Step 2: Execute the command
        raw_result = execute_generated_command(command)
        
        # Step 3: Format the result
        answer = format_result_with_gpt(query, raw_result)
        
        logging.info(f"Generated answer: {answer}")
        
        response = QueryResponse(query=query, answer=answer)
        return jsonify(response.dict())
    
    except ValidationError as e:
        logging.error(f"Validation Error: {e.errors()}")
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
