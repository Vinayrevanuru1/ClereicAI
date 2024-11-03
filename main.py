import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from kubernetes import client, config

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


@app.route('/query', methods=['POST'])
def create_query():
    try:
        # Extract the question from the request data
        request_data = request.json
        query = request_data.get('query')
        
        # Log the question
        logging.info(f"Received query: {query}")
        
        # Here, implement your logic to generate an answer for the given question.
        if "how many pods" in query.lower():
            pods = v1.list_namespaced_pod(namespace="test")
            answer = str(len(pods.items))
        elif "status of pod" in query.lower():
            pod_name = query.split("pod named")[-1].strip().strip("'").strip('"')
            try:
                pod = v1.read_namespaced_pod(name=pod_name, namespace="default")
                answer = pod.status.phase
            except client.exceptions.ApiException as e:
                answer = f"Error: {e.reason}"
        else:
            answer = "Query not understood."

        # Log the answer
        logging.info(f"Generated answer: {answer}")
        
        # Create the response model
        response = QueryResponse(query=query, answer=answer)
        
        return jsonify(response.dict())
    
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
