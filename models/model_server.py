from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------- LOAD MODELS ----------------

# Diabetes models
diabetes_log = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_logistic_regression.pkl"))
diabetes_rf = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_random_forest.pkl"))
diabetes_scaler = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_scaler.pkl"))


# Heart models
heart_knn = joblib.load(os.path.join(BASE_DIR, "heart/heart_knn_model.pkl"))
heart_svm = joblib.load(os.path.join(BASE_DIR, "heart/heart_svm_model.pkl"))
heart_scaler = joblib.load(os.path.join(BASE_DIR, "heart/heart_scaler.pkl"))


# Kidney models
kidney_nb = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_naive_bayes.pkl"))
kidney_dt = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_decision_tree.pkl"))
kidney_scaler = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_scaler.pkl"))


# ---------------- DIABETES ----------------

@app.route("/diabetes", methods=["POST"])
def diabetes():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "Missing features"}), 400

    X = np.array([data])
    X = diabetes_scaler.transform(X)

    return jsonify({
        "logistic_regression": int(diabetes_log.predict(X)[0]),
        "random_forest": int(diabetes_rf.predict(X)[0])
    })


# ---------------- HEART ----------------

@app.route("/heart", methods=["POST"])
def heart():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "Missing features"}), 400

    X = np.array([data])
    X = heart_scaler.transform(X)

    return jsonify({
        "knn": int(heart_knn.predict(X)[0]),
        "svm": int(heart_svm.predict(X)[0])
    })


# ---------------- KIDNEY ----------------

@app.route("/kidney", methods=["POST"])
def kidney():
    data = request.json.get("features")

    if not data:
        return jsonify({"error": "Missing features"}), 400

    X = np.array([data])
    X = kidney_scaler.transform(X)

    return jsonify({
        "naive_bayes": int(kidney_nb.predict(X)[0]),
        "decision_tree": int(kidney_dt.predict(X)[0])
    })


# ---------------- HEALTH CHECK ----------------

@app.route("/")
def home():
    return jsonify({
        "status": "Model Server Running",
        "endpoints": ["/diabetes", "/heart", "/kidney"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)