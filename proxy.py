from flask import Flask, request, Response
import requests

app = Flask(__name__)

ROUTES = {
    "agent-4b": "http://localhost:8000",
    "user-32b": "http://localhost:8001",
}

@app.route("/<path:path>", methods=["GET", "POST"])
def proxy(path):
    data = request.get_json(silent=True)
    model = data.get("model", "") if data else ""
    target = ROUTES.get(model, "http://localhost:8000")
    resp = requests.request(
        method=request.method,
        url=f"{target}/{path}",
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        json=data,
    )
    return Response(resp.content, status=resp.status_code,
                    content_type=resp.headers.get("content-type"))

if __name__ == "__main__":
    app.run(port=9000, threaded=True)
