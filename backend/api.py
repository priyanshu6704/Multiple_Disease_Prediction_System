from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

MODEL_URL = "http://model:8000"


@app.route("/predict/diabetes", methods=["POST"])
def diabetes():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "features missing"}), 400

    res = requests.post(f"{MODEL_URL}/diabetes", json={"features": data})
    return jsonify(res.json())


@app.route("/predict/heart", methods=["POST"])
def heart():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "features missing"}), 400

    res = requests.post(f"{MODEL_URL}/heart", json={"features": data})
    return jsonify(res.json())


@app.route("/predict/kidney", methods=["POST"])
def kidney():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "features missing"}), 400

    res = requests.post(f"{MODEL_URL}/kidney", json={"features": data})
    return jsonify(res.json())


@app.route("/")
def home():
    return jsonify({"status": "Backend API Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)