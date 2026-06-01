from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODELS ----------------

# Diabetes
diabetes_log = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_logistic_regression.pkl"))
diabetes_rf = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_random_forest.pkl"))
diabetes_scaler = joblib.load(os.path.join(BASE_DIR, "diabetes/diabetes_scaler.pkl"))

# Heart
heart_knn = joblib.load(os.path.join(BASE_DIR, "heart/heart_knn_model.pkl"))
heart_svm = joblib.load(os.path.join(BASE_DIR, "heart/heart_svm_model.pkl"))
heart_scaler = joblib.load(os.path.join(BASE_DIR, "heart/heart_scaler.pkl"))

# Kidney
kidney_nb = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_naive_bayes.pkl"))
kidney_dt = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_decision_tree.pkl"))
kidney_scaler = joblib.load(os.path.join(BASE_DIR, "kidney/kidney_scaler.pkl"))


# ---------------- SAFE HELPER ----------------

def safe_get_features():
    data = request.get_json()

    if not data or "features" not in data:
        return None, jsonify({"error": "Missing features"}), 400

    try:
        features = data["features"]
        X = np.array([features], dtype=float)
        return X, None, None

    except Exception as e:
        return None, jsonify({"error": str(e)}), 400


# ---------------- DIABETES ----------------

@app.route("/diabetes", methods=["POST"])
def diabetes():
    X, err, code = safe_get_features()
    if err:
        return err, code

    try:
        X = diabetes_scaler.transform(X)

        return jsonify({
            "logistic_regression": int(diabetes_log.predict(X)[0]),
            "random_forest": int(diabetes_rf.predict(X)[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- HEART ----------------

@app.route("/heart", methods=["POST"])
def heart():
    X, err, code = safe_get_features()
    if err:
        return err, code

    try:
        # FIX: prevents feature mismatch crash
        expected = heart_scaler.n_features_in_
        if X.shape[1] != expected:
            return jsonify({
                "error": f"Heart model expects {expected} features, got {X.shape[1]}"
            }), 400

        X = heart_scaler.transform(X)

        return jsonify({
            "knn": int(heart_knn.predict(X)[0]),
            "svm": int(heart_svm.predict(X)[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- KIDNEY ----------------

@app.route("/kidney", methods=["POST"])
def kidney():
    X, err, code = safe_get_features()
    if err:
        return err, code

    try:
        expected = kidney_scaler.n_features_in_
        if X.shape[1] != expected:
            return jsonify({
                "error": f"Kidney model expects {expected} features, got {X.shape[1]}"
            }), 400

        X = kidney_scaler.transform(X)

        return jsonify({
            "naive_bayes": int(kidney_nb.predict(X)[0]),
            "decision_tree": int(kidney_dt.predict(X)[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- HEALTH CHECK ----------------

@app.route("/")
def home():
    return jsonify({
        "status": "Model Server Running",
        "endpoints": ["/diabetes", "/heart", "/kidney"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)