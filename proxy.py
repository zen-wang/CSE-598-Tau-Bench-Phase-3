import os
from flask import Flask, request, Response
import requests

app = Flask(__name__)

# Configurable via env vars for dynamic port assignment (e.g., Gaudi jobs)
AGENT_URL = os.environ.get("VLLM_AGENT_URL", "http://localhost:8000")
USER_URL = os.environ.get("VLLM_USER_URL", "http://localhost:8001")

ROUTES = {
    "agent-4b": AGENT_URL,
    "agent-8b": AGENT_URL,
    "agent-14b": AGENT_URL,
    "agent-32b": AGENT_URL,
    "user-32b": USER_URL,
}

@app.route("/<path:path>", methods=["GET", "POST"])
def proxy(path):
    data = request.get_json(silent=True)
    model = data.get("model", "") if data else ""
    target = ROUTES.get(model, AGENT_URL)
    resp = requests.request(
        method=request.method,
        url=f"{target}/{path}",
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        json=data,
    )
    return Response(resp.content, status=resp.status_code,
                    content_type=resp.headers.get("content-type"))

if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "9090"))
    app.run(port=port, threaded=True)
