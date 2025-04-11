# from flask import Flask, request, jsonify
# import pickle  # Assuming you have a saved ML model

# app = Flask(_name_)

# # Load the trained Alzheimer's Risk Assessment model
# with open("alzheimer_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     # Convert received data into a feature list (ensure order matches training)
#     features = [
#         float(data["age"]), int(data["gender"]), int(data["ethnicity"]),
#         float(data["education"]), float(data["bmi"]), int(data["smoking"]),
#         float(data["alcohol"]), float(data["diet"]), float(data["sleep"]),
#         int(data["family_history"]), int(data["diabetes"]), int(data["depression"]),
#         int(data["head_injury"]), int(data["hypertension"]), float(data["systolic_bp"]),
#         float(data["diastolic_bp"]), float(data["cholesterol_ldl"]), float(data["cholesterol_hdl"]),
#         float(data["cholesterol_trig"]), float(data["mmse"]), float(data["adl"]),
#         float(data["behavior"])
#     ]

#     # Make prediction
#     prediction = model.predict([features])[0]  # Assuming it's a binary classification

#     return jsonify({"prediction": "High Risk" if prediction == 1 else "Low Risk"})

# if _name_ == "_main_":
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import xgboost as xgb  # Ensure this is imported at the top


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")  # Renders index.html as the homepage


@app.route("/risk-assessment")
def risk_assessment():
    # Renders Risk_Assessment.html
    return render_template("Risk_Assessment.html")


@app.route("/get-recommendations")
def get_recommendations():
    return render_template("recommendations.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/brain-games")
def brain_games():
    return render_template("brain_games.html")


with open("continued_xgb_model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric Inputs
        age = int(request.form["Age"])
        bmi = float(request.form["BMI"])
        alcohol = float(request.form["AlcoholConsumption"])
        physical_activity = float(request.form["PhysicalActivity"])
        diet = float(request.form["DietQuality"])
        sleep = float(request.form["SleepQuality"])
        systolic_bp = float(request.form["SystolicBP"])
        diastolic_bp = float(request.form["DiastolicBP"])
        cholesterol_total = float(request.form["CholesterolTotal"])
        cholesterol_ldl = float(request.form["CholesterolLDL"])
        cholesterol_hdl = float(request.form["CholesterolHDL"])
        cholesterol_trig = float(request.form["CholesterolTriglycerides"])
        mmse = float(request.form["MMSE"])
        functional = float(request.form["FunctionalAssessment"])
        adl = float(request.form["ADL"])
        behavior = float(request.form["BehavioralProblems"])

        # Categorical Inputs
        memory_complaints = int(request.form["MemoryComplaints"])

        # Construct feature array
        features = np.array([
            age, bmi, alcohol, physical_activity, diet, sleep,
            systolic_bp, diastolic_bp, cholesterol_total,
            cholesterol_ldl, cholesterol_hdl, cholesterol_trig,
            mmse, functional, adl, behavior, memory_complaints
        ])

        if np.isnan(features).any():
            return jsonify({"result": "Error: One or more input values are missing or invalid."})

        # Predict using model
        prob = model.predict_proba(features[np.newaxis, :])[0][1]

        # Assign risk level
        if prob < 0.33:
            prediction_text = "Low Risk of Alzheimer's"
        elif prob < 0.66:
            prediction_text = "Moderate Risk of Alzheimer's"
        else:
            prediction_text = "High Risk of Alzheimer's"

        return jsonify({
            "result": prediction_text,
            "probability": f"{prob:.4f}"
        })

    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)

