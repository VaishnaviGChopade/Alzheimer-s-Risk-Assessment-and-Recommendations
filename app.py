# from flask import Flask, request, jsonify
# import pickle  # Assuming you have a saved ML model

# app = Flask(__name__)

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

# if __name__ == "__main__":
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
    return render_template("Risk_Assessment.html")  # Renders Risk_Assessment.html
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/about")  
def about():
    return render_template("About.html") 
@app.route("/brain-games")
def brain_games():
    return render_template("brain_games.html")
@app.route("/get-recommendations")
def get_recommendations():
    return render_template("recommendations.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form["age"])
        education = float(request.form["education"])
        bmi = float(request.form["bmi"])
        alcohol = float(request.form["alcohol"])
        diet = float(request.form["diet"])
        sleep = float(request.form["sleep"])
        systolic_bp = float(request.form["systolic_bp"])
        diastolic_bp = float(request.form["diastolic_bp"])
        cholesterol_ldl = float(request.form["cholesterol_ldl"])
        cholesterol_hdl = float(request.form["cholesterol_hdl"])
        cholesterol_trig = float(request.form["cholesterol_trig"])
        mmse = float(request.form["mmse"])
        adl = float(request.form["adl"])
        behavior = float(request.form["behavior"])

        # Extract categorical values as integers
        gender = int(request.form["gender"])  # 0 = Male, 1 = Female
        ethnicity = int(request.form["ethnicity"])
        smoking = int(request.form["smoking"])
        family_history = int(request.form["family_history"])
        diabetes = int(request.form["diabetes"])
        depression = int(request.form["depression"])
        head_injury = int(request.form["head_injury"])
        hypertension = int(request.form["hypertension"])

     
        features = np.array([
    age, gender, ethnicity, education, bmi, smoking,
    alcohol, diet, sleep, family_history, diabetes, depression,
    head_injury, hypertension, systolic_bp, diastolic_bp,
    cholesterol_ldl, cholesterol_hdl, cholesterol_trig,
    mmse, adl, behavior
])

        if np.isnan(features).any():
          return jsonify({"result": "Error: One or more input values are invalid or missing."})

        feature_names = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "SystolicBP",
    "DiastolicBP", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    "MMSE", "ADL_FunctionalCombined", "Behavioral_Issues_Combined"
]

        dmat = xgb.DMatrix(features[np.newaxis, :], feature_names=feature_names)
        #prediction = model.predict(dmat)[0] 
        prediction_prob = model.predict(dmat)
        print(f"Prediction Probability: {prediction_prob}")
        if isinstance(prediction_prob, np.ndarray) and len(prediction_prob) == 1:
            prediction = int(prediction_prob[0] > 0.3299)  # Convert probability to 0 or 1
            prediction_text = "" if prediction == 0 else "High Risk of Alzheimer's"
            print(f"Prediction : {prediction_text}")
        else:
            prediction_text = "Error: Unexpected model output."
        if prediction == 0:  # Even if the model predicts "No Alzheimer's"
            if prediction_prob >= 0.2:
                risk_category = "Moderate Risk of having alzheimers"
                prediction_text+=risk_category
            elif prediction_prob >= 0.1:
                risk_category = "Low Risk of having alzheimers"
                prediction_text+=risk_category
            else:
                risk_category = "Very Low Risk of having alzheimers"
                prediction_text+=risk_category
        

        return jsonify({"result": prediction_text, "probability": str(prediction_prob[0])})

    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)


